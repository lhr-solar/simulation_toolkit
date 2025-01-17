import math
import numpy as np

from sim.system_models.vehicle_systems.vehicle_system_model import VehicleSystemModel

from sim.model_parameters.cars.car import Car
from sim.system_models.vectors.controls_vector import ControlsVector
from sim.system_models.vectors.observables_vector import ObservablesVector
from sim.system_models.vectors.state_vector import StateVector
from sim.system_models.vectors.state_dot_vector import StateDotVector
from sim.util.math.conversions import rpm_to_rads, rads_to_rpm


class PowertrainModel(VehicleSystemModel):
    def __init__(self):
        super().__init__()

        self.controls_in = [
            "torque_request",
            "cooling_percent",
        ]

        self.state_in = [
            "hv_battery_charge",
            "lv_battery_charge",

            "hv_battery_temperature",
            "inverter_temperature",
            "motor_temperature",
            "coolant_temperature",

            "motor_rpm",
            # "v_long",  # this will be for cooling from external airflow later
        ]

        self.state_out = [
            "hv_battery_current",
            "lv_battery_current",

            "hv_battery_net_heat",
            "inverter_net_heat",
            "motor_net_heat",
            "coolant_net_heat",

            "applied_torque_fl",
            "applied_torque_fr",
            "applied_torque_rl",
            "applied_torque_rr",
        ]

        self.observables = [
            "hv_battery_open_circuit_voltage",
            "hv_battery_terminal_voltage",
            "lv_battery_open_circuit_voltage",
            "lv_battery_terminal_voltage",
        ]

    # TODO multi-motor configurations
    # TODO cooling from airflow (function of vehicle velocity)
    # TODO diff?
    # TODO cooling from infrared radiation??

    def eval(self, car: Car, controls_in: ControlsVector, state_in: StateVector,
             state_out: StateDotVector, observables: ObservablesVector):

        lv_system_power_out = car.lv_system_constant_power_draw + car.cooling_power_draw(controls_in.cooling_percent)
        hv_battery_cooling, hv_battery_cooling_from_coolant = self._calculate_cooling(0, 0, 0)  # TODO implement
        inverter_cooling, inverter_cooling_from_coolant = self._calculate_cooling(0, 0, 0)  # TODO implement
        motor_cooling, motor_cooling_from_coolant = self._calculate_cooling(0, 0, 0)  # TODO implement
        coolant_cooling = 0  # TODO calculate coolant cooling as a function of fan output
        coolant_heating = hv_battery_cooling_from_coolant + inverter_cooling_from_coolant + motor_cooling_from_coolant

        hv_battery_open_circuit_voltage = car.hv_battery_open_circuit_voltage(state_in.hv_battery_charge)
        hv_battery_internal_resistance = (
            car.hv_battery_internal_resistance(state_in.hv_battery_charge))  # , state_in.hv_battery_temperature))

        motor_torque = 0
        motor_back_emf = state_in.motor_rpm / car.motor_induced_voltage
        motor_efficiency = car.motor_efficiency(motor_torque)

        observables.line_pressures = [0, 0]
        observables.motor_torque = 0

        observables.applied_torques = [0, 0, 0, 0]
        observables.regen_torques = [0, 0, 0, 0]
        observables.mechanical_brake_torque = [0, 0, 0, 0]

        motor_power_out = 0
        inverter_power_out = 0
        hv_battery_power_out = 0

        if controls_in.torque_request > 0:
            torque_request = controls_in.torque_request
            available_voltage = hv_battery_open_circuit_voltage - motor_back_emf
            available_current = available_voltage / (car.motor_winding_resistance + hv_battery_internal_resistance)
            rated_torque = car.motor_peak_torque(state_in.motor_rpm)

            possible_current = min(rated_torque / car.motor_kt, car.motor_peak_current, available_current)
            possible_power = car.hv_battery_nominal_voltage * possible_current
            available_torque = possible_power / rpm_to_rads(state_in.motor_rpm) if state_in.motor_rpm else 1e9

            motor_torque = min(torque_request, rated_torque, available_torque)
            observables.motor_torque = motor_torque

            diff_torque = motor_torque * car.gear_ratio * car.diff_efficiency
            observables.applied_torques = [
                0,
                0,
                diff_torque * car.drivetrain_efficiency / 2,  # TODO add diff lmao
                diff_torque * car.drivetrain_efficiency / 2,
            ]

            motor_power_out = motor_torque * rpm_to_rads(state_in.motor_rpm)
            inverter_power_out = motor_power_out / motor_efficiency
            hv_battery_power_out = inverter_power_out / car.inverter_efficiency

        elif controls_in.torque_request < 0 and car.regen_enabled:
            torque_request = controls_in.torque_request
            available_voltage = motor_back_emf
            available_current = available_voltage / (car.motor_winding_resistance + hv_battery_internal_resistance)
            rated_torque = car.motor_peak_torque(state_in.motor_rpm)

            possible_current = min(rated_torque / car.motor_kt, car.motor_peak_current, available_current)
            possible_power = car.hv_battery_nominal_voltage * possible_current
            available_power = min(possible_power, car.power_limit / car.inverter_efficiency / motor_efficiency)
            available_torque = available_power / rpm_to_rads(state_in.motor_rpm) if state_in.motor_rpm else 1e9

            motor_torque = -min(-torque_request, rated_torque, available_torque)
            observables.motor_torque = motor_torque

            diff_torque = motor_torque * car.gear_ratio * car.diff_efficiency
            observables.regen_torques = [
                0,
                0,
                diff_torque * car.drivetrain_efficiency / 2,
                diff_torque * car.drivetrain_efficiency / 2,
            ]

            motor_power_out = motor_torque * rpm_to_rads(state_in.motor_rpm)
            inverter_power_out = motor_power_out * motor_efficiency
            hv_battery_power_out = inverter_power_out * car.inverter_efficiency

        if controls_in.brake_pct > 0:
            max_driver_force = car.max_DF
            pedal_ratio = car.pedal_ratio
            brake_bias = car.brake_bias
            master_cylinder_SAs = car.MC_SA
            caliper_SAs = car.C_SA
            pad_friction_coefficients = car.mu
            effective_rotor_radii = car.eff_rotor_radius
            tire_radii = car.tire_radii

            brake_calcs = self._mech_brake_calcs(DF=max_driver_force, PR=pedal_ratio, BB=brake_bias,
                                                 MC_SA=master_cylinder_SAs,
                                                 C_SA=caliper_SAs, mu=pad_friction_coefficients,
                                                 RR=effective_rotor_radii,
                                                 TR=tire_radii, brake_pct=controls_in.brake_pct)

            observables.line_pressures = brake_calcs[0]
            observables.mechanical_brake_torque = brake_calcs[1]

        hv_battery_power_out += car.has_dcdc * (lv_system_power_out / car.dcdc_efficiency)
        lv_battery_power_out = (not car.has_dcdc) * lv_system_power_out

        hv_battery_current = self._calculate_battery_current(hv_battery_power_out, hv_battery_open_circuit_voltage,
                                                             hv_battery_internal_resistance)
        hv_battery_voltage_drop = hv_battery_internal_resistance * hv_battery_current
        hv_battery_terminal_voltage = hv_battery_open_circuit_voltage - hv_battery_voltage_drop

        lv_battery_open_circuit_voltage = car.lv_battery_open_circuit_voltage(state_in.lv_battery_charge)
        lv_battery_current = self._calculate_battery_current(lv_battery_power_out, lv_battery_open_circuit_voltage,
                                                             car.lv_battery_internal_resistance)
        lv_battery_voltage_drop = car.lv_battery_internal_resistance * lv_battery_current
        lv_battery_terminal_voltage = lv_battery_open_circuit_voltage - lv_battery_voltage_drop

        hv_battery_heat_loss = hv_battery_voltage_drop * hv_battery_current
        inverter_heat_loss = hv_battery_power_out - inverter_power_out
        motor_heat_loss = inverter_power_out - motor_power_out

        state_out.hv_battery_current = hv_battery_current
        state_out.lv_battery_current = lv_battery_current
        state_out.hv_battery_net_heat = hv_battery_heat_loss - hv_battery_cooling
        state_out.inverter_net_heat = inverter_heat_loss - inverter_cooling
        state_out.motor_net_heat = motor_heat_loss - motor_cooling
        state_out.coolant_net_heat = coolant_heating - coolant_cooling
        state_out.powertrain_torques = (np.array(observables.applied_torques) + np.array(observables.regen_torques)
                                        + np.array(observables.mechanical_brake_torque))

        observables.hv_battery_open_circuit_voltage = hv_battery_open_circuit_voltage
        observables.hv_battery_terminal_voltage = hv_battery_terminal_voltage
        observables.lv_battery_open_circuit_voltage = lv_battery_open_circuit_voltage
        observables.lv_battery_terminal_voltage = lv_battery_terminal_voltage
        observables.hv_battery_power_out = hv_battery_power_out
        observables.inverter_power_out = inverter_power_out
        observables.motor_power_out = motor_power_out
        # TODO add more observables from the existing variables

    def _calculate_battery_current(self, battery_power: float, battery_open_circuit_voltage: float,
                                   battery_internal_resistance: float) -> float:
        """
        solves for current given power output, open circuit voltage, and internal resistance

        i = p / v
        v = v_oc - i*r
        i = p / (v_oc - i*r)
        (-r)i^2 + (v_oc)i - (p) = 0

        quadratic equation

        i = (-v_oc +/- sqrt(v_oc**2 - 4*r*p)))/(-2r)
        i = (v_oc +/- sqrt(v_oc**2 - 4*r*p)))/(2r)

        we only want the smaller i, the larger one is extraneous
        """
        # TODO rethink this with regen, as power will be negative and some sign changes will be necessary

        discriminant = (battery_open_circuit_voltage * battery_open_circuit_voltage
                        - 4 * battery_internal_resistance * battery_power)
        if discriminant < 0:
            raise Exception("too much current! not sure what to do now. maybe try not flooring it bruh")
        i = (battery_open_circuit_voltage - math.sqrt(discriminant)) / 2 / battery_internal_resistance
        return i

    def _calculate_cooling(self, object_temp: float, coolant_temp: float, coolant_area: float) -> (float, float):
        # TODO cooling calculations!!
        return 0, 0

    # def _diff_bias_ratio(self, steered_angle, body_slip, diff_torque, wheel_angular_velocities, diff_radius, motor_radius):
    #     if steered_angle == 0 and body_slip == 0 or diff_torque == 0:
    #         return [0.5, 0.5]

    #     traction_bias = self.params.diff_fl + self.params.diff_preload/torque_on_diff

    #     if self.state.is_left_diff_bias:
    #         return np.array([traction_bias, 1 - traction_bias])
    #     else:
    #         return np.array([1 - traction_bias, traction_bias])

    def _mech_brake_calcs(self, DF, PR, BB, MC_SA, C_SA, mu, RR, TR, brake_pct):
        pedal_force = DF * brake_pct

        front_MC_SA = MC_SA[0]
        rear_MC_SA = MC_SA[1]
        front_C_SA = C_SA[0]
        rear_C_SA = C_SA[1]
        front_mu = mu[0]
        rear_mu = mu[1]
        front_RR = RR[0]
        rear_RR = RR[1]
        front_TR = TR[0]
        rear_TR = TR[2]

        front_pedal_force = pedal_force * PR * BB
        rear_pedal_force = pedal_force * PR * (1 - BB)
        front_line_pressure = front_pedal_force / front_MC_SA
        rear_line_pressure = rear_pedal_force / rear_MC_SA

        front_braking_force = front_line_pressure * front_C_SA * front_mu
        rear_braking_force = rear_line_pressure * rear_C_SA * rear_mu

        front_braking_torque = -1 * front_braking_force * front_RR
        rear_braking_torque = -1 * rear_braking_force * rear_RR

        line_pressures = [front_line_pressure, rear_line_pressure]
        braking_torques = [front_braking_torque, front_braking_torque, rear_braking_torque, rear_braking_torque]

        return [line_pressures, braking_torques]

    # def _torque_bias_ratio(self, steered_angle, body_slip, torque_on_diff):
    #     # if on a pure straight, diff doesnt bias. Otherwise it does. BREAKAWAY TORQUE BABY
    #     if steered_angle == 0 and body_slip == 0 or torque_on_diff == 0:
    #         return np.array([0.5, 0.5])

    #     traction_bias = self.params.diff_fl + self.params.diff_preload / torque_on_diff

    #     if self.state.is_left_diff_bias:
    #         return np.array([traction_bias, 1 - traction_bias])
    #     else:
    #         return np.array([1 - traction_bias, traction_bias])
