from vehicle_model.suspension_model.suspension_elements.tertiary_elements.double_wishbone import DoubleWishbone
from vehicle_model.suspension_model.suspension_elements.quinary_elements.full_suspension import FullSuspension
from vehicle_model.suspension_model.suspension_elements.quaternary_elements.axle import Axle
from vehicle_model.suspension_model.suspension_elements.secondary_elements.cg import CG
from vehicle_model._assets.pickle_helpers import pickle_import
from matplotlib.backends.backend_pdf import PdfPages
from vehicle_model._assets.interp import interp4d
from typing import Callable, Sequence, Tuple
from matplotlib.figure import Figure
from collections import OrderedDict
from dash import Dash, dash_table
import matplotlib.pyplot as plt
import pyvista as pv
import pandas as pd
import numpy as np
import pickle
import os


class SuspensionModel:
    """
    ## Suspension Model

    Designed to model kinematics and force-based properties

    ###### Note, all conventions comply with SAE-J670 Z-up

    Parameters
    ----------
    FL_inboard_points : Sequence[Sequence[float]]
        Array containing all inboard coordinates of the front-left suspension assembly
        - Order: [[Upper Fore], [Upper Aft], [Lower Fore], [Lower Aft], [Tie Rod], [Push / Pull Rod]]
    FL_outboard_points : Sequence[Sequence[float]]
        Array containing all outboard coordinates of the front-left suspension assembly
        - Order: [[Upper Fore], [Upper Aft], [Lower Fore], [Lower Aft], [Tie Rod], [Push / Pull Rod]]
    FL_bellcrank_params : Sequence[Sequence[float]]
        Relevant bellcrank to shock parameters for front-left suspension assembly
        - Order: [[Pivot], [Pivot Direction], [Shock Outboard], [Shock Inboard]]
    FL_upper : bool
        True if push/pull rod mounts to upper wishbone
    FL_contact_patch : Sequence[float]
        Coordinates of the front-left contact patch
    FL_inclination_angle : float
        Inclination angle of the front-left tire in degrees
    FL_toe : float
        Static toe angle of the front-left tire in degrees
        - Uses same sign convention as slip angle, NOT symmetric
    FL_rate : float
        Spring rate of the front left assembly in N/m
    FL_weight : float
        Weight of the front left vehicle corner in N

    ---

    FR_inboard_points : Sequence[Sequence[float]]
        Array containing all inboard coordinates of the front-right suspension assembly
        - Order: [[Upper Fore], [Upper Aft], [Lower Fore], [Lower Aft], [Tie Rod], [Push / Pull Rod]]
    FR_outboard_points : Sequence[Sequence[float]]
        Array containing all outboard coordinates of the front-right suspension assembly
        - Order: [[Upper Fore], [Upper Aft], [Lower Fore], [Lower Aft], [Tie Rod], [Push / Pull Rod]]
    FR_bellcrank_params : Sequence[Sequence[float]]
        Relevant bellcrank to shock parameters for front-right suspension assembly
        - Order: [[Pivot], [Pivot Direction], [Shock Outboard], [Shock Inboard]]
    FR_upper : bool
        True if push/pull rod mounts to upper wishbone
    FR_contact_patch : Sequence[float]
        Coordinates of the front-right contact patch
    FR_inclination_angle : float
        Inclination angle of the front-right tire in degrees
    FR_toe : float
        Static toe angle of the front-right tire in degrees
        - Uses same sign convention as slip angle, NOT symmetric
    FR_rate : float
        Spring rate of the front right assembly in N/m
    FR_weight : float
        Weight of the front right vehicle corner in N
    
    ---

    RL_inboard_points : Sequence[Sequence[float]]
        Array containing all inboard coordinates of the rear-left suspension assembly
        - Order: [[Upper Fore], [Upper Aft], [Lower Fore], [Lower Aft], [Tie Rod], [Push / Pull Rod]]
    RL_outboard_points : Sequence[Sequence[float]]
        Array containing all outboard coordinates of the rear-left suspension assembly
        - Order: [[Upper Fore], [Upper Aft], [Lower Fore], [Lower Aft], [Tie Rod], [Push / Pull Rod]]
    RL_bellcrank_params : Sequence[Sequence[float]]
        Relevant bellcrank to shock parameters for rear-left suspension assembly
        - Order: [[Pivot], [Pivot Direction], [Shock Outboard], [Shock Inboard]]
    RL_upper : bool
        True if push/pull rod mounts to upper wishbone
    RL_contact_patch : Sequence[float]
        Coordinates of the rear-left contact patch
    RL_inclination_angle : float
        Inclination angle of the rear-left tire in degrees
    RL_toe : float
        Static toe angle of the rear-left tire in degrees
        - Uses same sign convention as slip angle, NOT symmetric
    RL_rate : float
        Spring rate of the rear left assembly in N/m
    RL_weight : float
        Weight of the rear left vehicle corner in N
    
    --- 

    RR_inboard_points : Sequence[Sequence[float]]
        Array containing all inboard coordinates of the rear-right suspension assembly
        - Order: [[Upper Fore], [Upper Aft], [Lower Fore], [Lower Aft], [Tie Rod], [Push / Pull Rod]]
    RR_outboard_points : Sequence[Sequence[float]]
        Array containing all outboard coordinates of the rear-right suspension assembly
        - Order: [[Upper Fore], [Upper Aft], [Lower Fore], [Lower Aft], [Tie Rod], [Push / Pull Rod]]
    RR_bellcrank_params : Sequence[Sequence[float]]
        Relevant bellcrank to shock parameters for rear-right suspension assembly
        - Order: [[Pivot], [Pivot Direction], [Shock Outboard], [Shock Inboard]]
    RR_upper : bool
        True if push/pull rod mounts to upper wishbone
    RR_contact_patch : Sequence[float]
        Coordinates of the rear-right contact patch
    RR_inclination_angle : float
        Inclination angle of the rear-right tire in degrees
    RR_toe : float
        Static toe angle of the rear-right tire in degrees
        - Uses same sign convention as slip angle, NOT symmetric
    RR_rate : float
        Spring rate of the rear right assembly in N/m
    RR_weight : float
        Weight of the rear right vehicle corner in N
    
    ---

    tire_radius : float
        Radius of tire
        - Must use same units as suspension coordinates above
    tire_width : float
        Width of tire
        - Must use same units as suspension coordinates above
    cg_location : Sequence[float]
        Coordinates of the center of gravity
        - Must use same units as suspension coordinates above
    show_ICs : bool
        Toggle visibility of double wishbones vs swing arms
    plotter : pv.Plotter
        Plotting object
        - Import using ```from vehicle_model.suspension_model.assets.plotter import Plotter```
        - Define plotter object using ```plotter = Plotter()```
    """
    def __init__(
            self,

            FL_inboard_points: Sequence[Sequence[float]],
            FL_outboard_points: Sequence[Sequence[float]],
            FL_bellcrank_params: Sequence[Sequence[float]],
            FL_upper: bool,
            FL_contact_patch: Sequence[float],
            FL_inclination_angle: float,
            FL_toe: float,
            FL_rate: float,
            FL_weight: float,

            FR_inboard_points: Sequence[Sequence[float]],
            FR_outboard_points: Sequence[Sequence[float]],
            FR_bellcrank_params: Sequence[Sequence[float]],
            FR_upper: bool,
            FR_contact_patch: Sequence[float],
            FR_inclination_angle: float,
            FR_toe: float,
            FR_rate: float,
            FR_weight: float,

            Fr_ARBK: float,
            Fr_ARBMR: float,

            RL_inboard_points: Sequence[Sequence[float]],
            RL_outboard_points: Sequence[Sequence[float]],
            RL_bellcrank_params: Sequence[Sequence[float]],
            RL_upper: bool,
            RL_contact_patch: Sequence[float],
            RL_inclination_angle: float,
            RL_toe: float,
            RL_rate: float,
            RL_weight: float,

            RR_inboard_points: Sequence[Sequence[float]],
            RR_outboard_points: Sequence[Sequence[float]],
            RR_bellcrank_params: Sequence[Sequence[float]],
            RR_upper: bool,
            RR_contact_patch: Sequence[float],
            RR_inclination_angle: float,
            RR_toe: float,
            RR_rate: float,
            RR_weight: float,

            Rr_ARBK: float,
            Rr_ARBMR: float,

            tire_radius: float,
            tire_width: float,

            cg_location: Sequence[float],
            show_ICs: bool,
            
            plotter: pv.Plotter) -> None:
            
        self.parameters = locals().copy()
        del self.parameters['self']
        
        # Initialize plotter
        self.plotter = plotter
        self.verbose: bool = False

        # Initialize state
        self.current_heave = 0
        self.current_pitch = 0
        self.current_roll = 0

        # CG location
        self.cg: CG = CG(position=cg_location)

        # Initialize each corner assembly
        self.FL_double_wishbone: DoubleWishbone = DoubleWishbone(inboard_points=FL_inboard_points,
                                                                outboard_points=FL_outboard_points,
                                                                bellcrank_params=FL_bellcrank_params,
                                                                spring_rate=FL_rate,
                                                                weight=FL_weight,
                                                                cg=self.cg,
                                                                upper=FL_upper,
                                                                contact_patch=FL_contact_patch,
                                                                inclination_angle=FL_inclination_angle * np.pi / 180,
                                                                toe=FL_toe * np.pi / 180,
                                                                tire_radius=tire_radius,
                                                                tire_width=tire_width,
                                                                show_ICs=show_ICs)
            
        self.FR_double_wishbone: DoubleWishbone = DoubleWishbone(inboard_points=FR_inboard_points,
                                                                outboard_points=FR_outboard_points,
                                                                bellcrank_params=FR_bellcrank_params,
                                                                spring_rate=FR_rate,
                                                                weight=FR_weight,
                                                                cg=self.cg,
                                                                upper=FR_upper,
                                                                contact_patch=FR_contact_patch,
                                                                inclination_angle=FR_inclination_angle * np.pi / 180,
                                                                toe=FR_toe * np.pi / 180,
                                                                tire_radius=tire_radius,
                                                                tire_width=tire_width,
                                                                show_ICs=show_ICs)
            
        self.RL_double_wishbone: DoubleWishbone = DoubleWishbone(inboard_points=RL_inboard_points,
                                                                outboard_points=RL_outboard_points,
                                                                bellcrank_params=RL_bellcrank_params,
                                                                spring_rate=RL_rate,
                                                                weight=RL_weight,
                                                                cg=self.cg,
                                                                upper=RL_upper,
                                                                contact_patch=RL_contact_patch,
                                                                inclination_angle=RL_inclination_angle * np.pi / 180,
                                                                toe=RL_toe * np.pi / 180,
                                                                tire_radius=tire_radius,
                                                                tire_width=tire_width,
                                                                show_ICs=show_ICs)
            
        self.RR_double_wishbone: DoubleWishbone = DoubleWishbone(inboard_points=RR_inboard_points,
                                                                outboard_points=RR_outboard_points,
                                                                bellcrank_params=RR_bellcrank_params,
                                                                spring_rate=RR_rate,
                                                                weight=RR_weight,
                                                                cg=self.cg,
                                                                upper=RR_upper,
                                                                contact_patch=RR_contact_patch,
                                                                inclination_angle=RR_inclination_angle * np.pi / 180,
                                                                toe=RR_toe * np.pi / 180,
                                                                tire_radius=tire_radius,
                                                                tire_width=tire_width,
                                                                show_ICs=show_ICs)

        # Initialize each axle
        self.Fr_axle: Axle = Axle(left_assy=self.FL_double_wishbone, right_assy=self.FR_double_wishbone, cg=self.cg)
        self.Rr_axle: Axle = Axle(left_assy=self.RL_double_wishbone, right_assy=self.RR_double_wishbone, cg=self.cg)
        
        # Initialize front and rear axles together
        self.full_suspension: FullSuspension = FullSuspension(Fr_axle=self.Fr_axle, Rr_axle=self.Rr_axle, cg=self.cg)

        # Elements for plotting
        self.elements = [self.full_suspension]

    def generate_kin(self) -> None:
        """
        ## Generate Kinematics

        Generates and caches vehicle kinematics

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.FL_dw = self.full_suspension.Fr_axle.left
        self.FR_dw = self.full_suspension.Fr_axle.right
        self.RL_dw = self.full_suspension.Rr_axle.left
        self.RR_dw = self.full_suspension.Rr_axle.right
        corners = [self.FL_dw, self.FR_dw, self.RL_dw, self.RR_dw]
        if os.listdir("./outputs/kin_interp"):
            print("Kin lookup object found\n")
            file_path = "./outputs/kin_interp/" + os.listdir("./outputs/kin_interp")[0]
            with open(file_path, 'rb') as inp:
                lookup_objs = pickle.load(inp)

            self.FL_gamma_lookup = lookup_objs[0]
            self.FL_toe_lookup = lookup_objs[1]
            self.FL_caster_lookup = lookup_objs[2]
            self.FL_cpx_lookup = lookup_objs[3]
            self.FL_cpy_lookup = lookup_objs[4]
            self.FL_cpz_lookup = lookup_objs[5]
            self.FL_jounce_lookup = lookup_objs[6]
            self.FL_Kw_lookup = lookup_objs[7]
            self.FL_FV_FAPx_lookup = lookup_objs[8]
            self.FL_FV_FAPy_lookup = lookup_objs[9]
            self.FL_FV_FAPz_lookup = lookup_objs[10]
            self.FL_SV_FAPx_lookup = lookup_objs[11]
            self.FL_SV_FAPy_lookup = lookup_objs[12]
            self.FL_SV_FAPz_lookup = lookup_objs[13]
            self.FL_FV_ICx_lookup = lookup_objs[14]
            self.FL_FV_ICy_lookup = lookup_objs[15]
            self.FL_FV_ICz_lookup = lookup_objs[16]
            self.FL_SV_ICx_lookup = lookup_objs[17]
            self.FL_SV_ICy_lookup = lookup_objs[18]
            self.FL_SV_ICz_lookup = lookup_objs[19]

            self.FR_gamma_lookup = lookup_objs[20]
            self.FR_toe_lookup = lookup_objs[21]
            self.FR_caster_lookup = lookup_objs[22]
            self.FR_cpx_lookup = lookup_objs[23]
            self.FR_cpy_lookup = lookup_objs[24]
            self.FR_cpz_lookup = lookup_objs[25]
            self.FR_jounce_lookup = lookup_objs[26]
            self.FR_Kw_lookup = lookup_objs[27]
            self.FR_FV_FAPx_lookup = lookup_objs[28]
            self.FR_FV_FAPy_lookup = lookup_objs[29]
            self.FR_FV_FAPz_lookup = lookup_objs[30]
            self.FR_SV_FAPx_lookup = lookup_objs[31]
            self.FR_SV_FAPy_lookup = lookup_objs[32]
            self.FR_SV_FAPz_lookup = lookup_objs[33]
            self.FR_FV_ICx_lookup = lookup_objs[34]
            self.FR_FV_ICy_lookup = lookup_objs[35]
            self.FR_FV_ICz_lookup = lookup_objs[36]
            self.FR_SV_ICx_lookup = lookup_objs[37]
            self.FR_SV_ICy_lookup = lookup_objs[38]
            self.FR_SV_ICz_lookup = lookup_objs[39]
            
            self.RL_gamma_lookup = lookup_objs[40]
            self.RL_toe_lookup = lookup_objs[41]
            self.RL_caster_lookup = lookup_objs[42]
            self.RL_cpx_lookup = lookup_objs[43]
            self.RL_cpy_lookup = lookup_objs[44]
            self.RL_cpz_lookup = lookup_objs[45]
            self.RL_jounce_lookup = lookup_objs[46]
            self.RL_Kw_lookup = lookup_objs[47]
            self.RL_FV_FAPx_lookup = lookup_objs[48]
            self.RL_FV_FAPy_lookup = lookup_objs[49]
            self.RL_FV_FAPz_lookup = lookup_objs[50]
            self.RL_SV_FAPx_lookup = lookup_objs[51]
            self.RL_SV_FAPy_lookup = lookup_objs[52]
            self.RL_SV_FAPz_lookup = lookup_objs[53]
            self.RL_FV_ICx_lookup = lookup_objs[54]
            self.RL_FV_ICy_lookup = lookup_objs[55]
            self.RL_FV_ICz_lookup = lookup_objs[56]
            self.RL_SV_ICx_lookup = lookup_objs[57]
            self.RL_SV_ICy_lookup = lookup_objs[58]
            self.RL_SV_ICz_lookup = lookup_objs[59]

            self.RR_gamma_lookup = lookup_objs[60]
            self.RR_toe_lookup = lookup_objs[61]
            self.RR_caster_lookup = lookup_objs[62]
            self.RR_cpx_lookup = lookup_objs[63]
            self.RR_cpy_lookup = lookup_objs[64]
            self.RR_cpz_lookup = lookup_objs[65]
            self.RR_jounce_lookup = lookup_objs[66]
            self.RR_Kw_lookup = lookup_objs[67]
            self.RR_FV_FAPx_lookup = lookup_objs[68]
            self.RR_FV_FAPy_lookup = lookup_objs[69]
            self.RR_FV_FAPz_lookup = lookup_objs[70]
            self.RR_SV_FAPx_lookup = lookup_objs[71]
            self.RR_SV_FAPy_lookup = lookup_objs[72]
            self.RR_SV_FAPz_lookup = lookup_objs[73]
            self.RR_FV_ICx_lookup = lookup_objs[74]
            self.RR_FV_ICy_lookup = lookup_objs[75]
            self.RR_FV_ICz_lookup = lookup_objs[76]
            self.RR_SV_ICx_lookup = lookup_objs[77]
            self.RR_SV_ICy_lookup = lookup_objs[78]
            self.RR_SV_ICz_lookup = lookup_objs[79]

            self.cgx_lookup = lookup_objs[80]
            self.cgy_lookup = lookup_objs[81]
            self.cgz_lookup = lookup_objs[82]

            self.Fr_Kr_lookup = lookup_objs[83]
            self.Rr_Kr_lookup = lookup_objs[84]
            self.Kp_lookup = lookup_objs[85]
        
        else: 
            print("Kin lookup object NOT found | Generating now:\n")
            refinement = 5
            steer_sweep = np.linspace(-1.75, 1.75, refinement) * 0.0254
            heave_sweep = np.linspace(-5, 5, refinement) * 0.0254
            pitch_sweep = np.linspace(-5, 5, refinement) * np.pi / 180
            roll_sweep = np.linspace(-5, 5, refinement) * np.pi / 180
        
            gamma_angles = {"FL": [], "FR": [], "RL": [], "RR": []}
            toe_angles = {"FL": [], "FR": [], "RL": [], "RR": []}
            caster_angles = {"FL": [], "FR": [], "RL": [], "RR": []}
            cp_locations = {"FL": {"x": [], "y": [], "z": []},
                            "FR": {"x": [], "y": [], "z": []},
                            "RL": {"x": [], "y": [], "z": []},
                            "RR": {"x": [], "y": [], "z": []}}
            FV_FAP_pos = {"FL": {"x": [], "y": [], "z": []},
                          "FR": {"x": [], "y": [], "z": []},
                          "RL": {"x": [], "y": [], "z": []},
                          "RR": {"x": [], "y": [], "z": []}}
            SV_FAP_pos = {"FL": {"x": [], "y": [], "z": []},
                          "FR": {"x": [], "y": [], "z": []},
                          "RL": {"x": [], "y": [], "z": []},
                          "RR": {"x": [], "y": [], "z": []}}
            FV_IC_pos = {"FL": {"x": [], "y": [], "z": []},
                         "FR": {"x": [], "y": [], "z": []},
                         "RL": {"x": [], "y": [], "z": []},
                         "RR": {"x": [], "y": [], "z": []}}
            SV_IC_pos = {"FL": {"x": [], "y": [], "z": []},
                         "FR": {"x": [], "y": [], "z": []},
                         "RL": {"x": [], "y": [], "z": []},
                         "RR": {"x": [], "y": [], "z": []}}
            jounce_vals = {"FL": [],
                           "FR": [],
                           "RL": [],
                           "RR": []}
            wheelrates = {"FL": [],
                          "FR": [],
                          "RL": [],
                          "RR": []}
            
            roll_stiffness = {"Fr": [], "Rr": []}
            roll_center = {"Fr": [], "Rr": []}
            pitch_stiffnesses = []
            pitch_centers = []
            cg_locs = {"x": [],
                       "y": [],
                       "z": []}
            
            labels = ["FL", "FR", "RL", "RR"]
            
            total_iter = refinement**4
            counter = 0
            
            for steer in steer_sweep:
                # self.steer(rack_displacement=steer)
                for heave in heave_sweep:
                    # self.heave(heave=heave)
                    for pitch in pitch_sweep:
                        # self.pitch(pitch=pitch * 180 / np.pi)
                        for roll in roll_sweep:
                            # self.full_suspension.hard_reset()
                            # self.self = SuspensionModel(**self.parameters)
                            # self.roll(roll=roll * 180 / np.pi)
                            # self.full_suspension.reset_position()
                            self.generate_kin_helper(steer=steer, heave=heave, pitch=pitch, roll=roll)
                            # self.full_suspension.flatten()
                            print(f"Creating Kin Lookup: {round(counter / total_iter * 100, 4)}%", end="\r")
                            
                            for i in range(len(corners)):
                                gamma_angles[labels[i]].append(corners[i].inclination_angle + roll)
                                toe_angles[labels[i]].append(corners[i].toe)
                                caster_angles[labels[i]].append(corners[i].caster)
                                cp_locations[labels[i]]["x"].append(corners[i].contact_patch.position[0])
                                cp_locations[labels[i]]["y"].append(corners[i].contact_patch.position[1])
                                cp_locations[labels[i]]["z"].append(corners[i].contact_patch.position[2])
                                FV_FAP_pos[labels[i]]["x"].append(corners[i].FV_FAP.position[0])
                                FV_FAP_pos[labels[i]]["y"].append(corners[i].FV_FAP.position[1])
                                FV_FAP_pos[labels[i]]["z"].append(corners[i].FV_FAP.position[2])
                                SV_FAP_pos[labels[i]]["x"].append(corners[i].SV_FAP.position[0])
                                SV_FAP_pos[labels[i]]["y"].append(corners[i].SV_FAP.position[1])
                                SV_FAP_pos[labels[i]]["z"].append(corners[i].SV_FAP.position[2])

                                FV_IC_pos[labels[i]]["x"].append(corners[i].FVIC.position[0])
                                FV_IC_pos[labels[i]]["y"].append(corners[i].FVIC.position[1])
                                FV_IC_pos[labels[i]]["z"].append(corners[i].FVIC.position[2])
                                SV_IC_pos[labels[i]]["x"].append(corners[i].SVIC.position[0])
                                SV_IC_pos[labels[i]]["y"].append(corners[i].SVIC.position[1])
                                SV_IC_pos[labels[i]]["z"].append(corners[i].SVIC.position[2])

                                jounce_vals[labels[i]].append(corners[i].total_jounce)
                                wheelrates[labels[i]].append(corners[i].wheelrate)
                            
                            self.generate_kin_helper(steer=0.000001, heave=heave, pitch=pitch, roll=roll)
                            roll_stiffness["Fr"].append(self.full_suspension.Fr_axle.roll_stiffness)
                            roll_center["Fr"].append(self.full_suspension.Fr_axle.kin_RC.true_KinRC.position)
                            roll_stiffness["Rr"].append(self.full_suspension.Rr_axle.roll_stiffness)
                            roll_center["Rr"].append(self.full_suspension.Rr_axle.kin_RC.true_KinRC.position)
                            
                            pitch_stiffnesses.append(self.full_suspension.pitch_stiffness)
                            pitch_centers.append(self.full_suspension.right_kin_PC.true_KinPC.position)
                            cg_locs["x"].append(self.full_suspension.cg.position[0])
                            cg_locs["y"].append(self.full_suspension.cg.position[1])
                            cg_locs["z"].append(self.full_suspension.cg.position[2])

                            counter += 1

            lookup_objs = []
            for label in labels:
                lookup_objs.append(interp4d(x=steer_sweep, y=heave_sweep, z=pitch_sweep, w=roll_sweep, v=gamma_angles[label]))
                lookup_objs.append(interp4d(x=steer_sweep, y=heave_sweep, z=pitch_sweep, w=roll_sweep, v=toe_angles[label]))
                lookup_objs.append(interp4d(x=steer_sweep, y=heave_sweep, z=pitch_sweep, w=roll_sweep, v=caster_angles[label]))
                lookup_objs.append(interp4d(x=steer_sweep, y=heave_sweep, z=pitch_sweep, w=roll_sweep, v=cp_locations[label]["x"]))
                lookup_objs.append(interp4d(x=steer_sweep, y=heave_sweep, z=pitch_sweep, w=roll_sweep, v=cp_locations[label]["y"]))
                lookup_objs.append(interp4d(x=steer_sweep, y=heave_sweep, z=pitch_sweep, w=roll_sweep, v=cp_locations[label]["z"]))
                lookup_objs.append(interp4d(x=steer_sweep, y=heave_sweep, z=pitch_sweep, w=roll_sweep, v=jounce_vals[label]))
                lookup_objs.append(interp4d(x=steer_sweep, y=heave_sweep, z=pitch_sweep, w=roll_sweep, v=wheelrates[label]))
                lookup_objs.append(interp4d(x=steer_sweep, y=heave_sweep, z=pitch_sweep, w=roll_sweep, v=FV_FAP_pos[label]["x"]))
                lookup_objs.append(interp4d(x=steer_sweep, y=heave_sweep, z=pitch_sweep, w=roll_sweep, v=FV_FAP_pos[label]["y"]))
                lookup_objs.append(interp4d(x=steer_sweep, y=heave_sweep, z=pitch_sweep, w=roll_sweep, v=FV_FAP_pos[label]["z"]))
                lookup_objs.append(interp4d(x=steer_sweep, y=heave_sweep, z=pitch_sweep, w=roll_sweep, v=SV_FAP_pos[label]["x"]))
                lookup_objs.append(interp4d(x=steer_sweep, y=heave_sweep, z=pitch_sweep, w=roll_sweep, v=SV_FAP_pos[label]["y"]))
                lookup_objs.append(interp4d(x=steer_sweep, y=heave_sweep, z=pitch_sweep, w=roll_sweep, v=SV_FAP_pos[label]["z"]))
                lookup_objs.append(interp4d(x=steer_sweep, y=heave_sweep, z=pitch_sweep, w=roll_sweep, v=FV_IC_pos[label]["x"]))
                lookup_objs.append(interp4d(x=steer_sweep, y=heave_sweep, z=pitch_sweep, w=roll_sweep, v=FV_IC_pos[label]["y"]))
                lookup_objs.append(interp4d(x=steer_sweep, y=heave_sweep, z=pitch_sweep, w=roll_sweep, v=FV_IC_pos[label]["z"]))
                lookup_objs.append(interp4d(x=steer_sweep, y=heave_sweep, z=pitch_sweep, w=roll_sweep, v=SV_IC_pos[label]["x"]))
                lookup_objs.append(interp4d(x=steer_sweep, y=heave_sweep, z=pitch_sweep, w=roll_sweep, v=SV_IC_pos[label]["y"]))
                lookup_objs.append(interp4d(x=steer_sweep, y=heave_sweep, z=pitch_sweep, w=roll_sweep, v=SV_IC_pos[label]["z"]))
            
            lookup_objs.append(interp4d(x=steer_sweep, y=heave_sweep, z=pitch_sweep, w=roll_sweep, v=cg_locs["x"]))
            lookup_objs.append(interp4d(x=steer_sweep, y=heave_sweep, z=pitch_sweep, w=roll_sweep, v=cg_locs["y"]))
            lookup_objs.append(interp4d(x=steer_sweep, y=heave_sweep, z=pitch_sweep, w=roll_sweep, v=cg_locs["z"]))
            lookup_objs.append(interp4d(x=steer_sweep, y=heave_sweep, z=pitch_sweep, w=roll_sweep, v=roll_stiffness["Fr"]))
            lookup_objs.append(interp4d(x=steer_sweep, y=heave_sweep, z=pitch_sweep, w=roll_sweep, v=roll_stiffness["Rr"]))
            lookup_objs.append(interp4d(x=steer_sweep, y=heave_sweep, z=pitch_sweep, w=roll_sweep, v=pitch_stiffnesses))

            with open(f"./outputs/kin_interp/kin_lookup.pkl", 'wb') as outp:
                pickle.dump(lookup_objs, outp, pickle.HIGHEST_PROTOCOL)
            
            # Re-initialize suspension with the cached model
            print()
            self.generate_kin()
            print()
    
    def steer(self, rack_displacement: float) -> None:
        """
        ## Steer

        Steer front suspension axle (capable of supporting rear steer as well)

        Parameters
        ----------
        rack_displacement : float
            Lateral translation of steering rack
            - Must use same units as suspension coordinates

        Returns
        ----------
        None
        """
        self.full_suspension.steer(rack_displacement=rack_displacement)
    
    def heave(self, heave: float) -> None:
        """
        ## Heave

        Heave front and rear suspension axles

        Parameters
        ----------
        heave : float
            Vertical translation of sprung mass
            - Must use same units as suspension coordinates above
            - Down is positive

        Returns
        ----------
        None
        """
        self.current_heave = heave
        self.full_suspension.heave(heave=heave)
    
    def pitch(self, pitch: float) -> None:
        """
        ## Pitch

        Pitch front and rear suspension axles

        Parameters
        ----------
        pitch : float
            Vehicle pitch in degrees
        
        Returns
        ----------
        None
        """
        self.current_pitch = pitch
        self.full_suspension.pitch(angle=pitch * np.pi / 180)
    
    def roll(self, roll: float) -> None:
        """
        ## Roll

        Roll front and rear suspension axles

        Parameters
        ----------
        roll : float
            Vehicle roll in degrees
        
        Returns
        ----------
        None
        """
        self.current_roll = roll
        self.full_suspension.roll(angle=roll * np.pi / 180)

    def plot_elements(self, plotter: pv.Plotter, verbose: bool = False, show_grid: bool = False) -> None:
        """
        ## Plot Elements

        Plots all children of full suspension model

        Parameters
        ----------
        plotter : py.Plotter
            Plotting object
        verbose : bool, optional
            Toggles visibility of roll/pitch center and cg, by default False
        show_grid : bool, optional
            Toggles grid visibility, by default False
        
        Returns
        ----------
        None
        """
        self.verbose = verbose
        plotter.clear()

        if show_grid:
            plotter.show_grid()

        plotter.add_ground(FL_cp=self.FL_double_wishbone.contact_patch, RL_cp=self.RL_double_wishbone.contact_patch, tire=self.FL_double_wishbone.tire)
        for element in self.elements:
            element.plot_elements(plotter=plotter, verbose=self.verbose)

    def add_slider(self, func: Callable, title: str, bounds: Tuple[float, float], pos: Tuple[Tuple[float, float], Tuple[float, float]]) -> None:
        """
        ## Add Slider

        Adds slider to Plotter window

        Parameters
        ----------
        func : Callable
            Function to execute slider values on
        title : str
            Title of slider
        bounds : Tuple[float, float]
            Limits of slider value
        pos : Tuple[Tuple[float, float], Tuple[float, float]]
            Position of slider (origin at lower left corner of window)
            - Start coordinate [Ratio X, Ratio Y]
            - End Coordinate [Ratio X, Ratio Y]
            - Max coordinate: [1, 1]
            - Min coordinate: [0, 0]
        
        Returns
        ----------
        None
        """
        self.plotter.add_slider(func, title, bounds, pos)

    def steer_slider(self, steer: float) -> None:
        """
        ## Steer Slider

        Steer slider function call

        Parameters
        ----------
        steer : float
            Rack displacement
            - Must use same units as suspension coordinates above

        Returns
        ----------
        None
        """
        self.steer(rack_displacement=steer * 3.50 / 360 * 0.0254)
        self.plot_elements(plotter=self.plotter, verbose=self.verbose)
    
    def heave_slider(self, heave: float) -> None:
        """
        ## Heave Slider

        Heave slider function call

        Parameters
        ----------
        heave : float
            Vertical translation of sprung mass
            - Must use same units as suspension coordinates above
            - Down is positive

        Returns
        ----------
        None
        """
        self.heave(heave=heave * 0.0254)
        self.plot_elements(plotter=self.plotter, verbose=self.verbose)

    def pitch_slider(self, pitch: float) -> None:
        """
        ## Pitch Slider

        Pitch slider function call

        Parameters
        ----------
        pitch : float
            Vehicle pitch in degrees

        Returns
        ----------
        None
        """
        self.pitch(pitch=pitch)
        self.plot_elements(plotter=self.plotter, verbose=self.verbose)

    def roll_slider(self, roll: float) -> None:
        """
        ## Roll Slider

        Roll slider function call

        Parameters
        ----------
        roll : float
            Vehicle roll in degrees

        Returns
        ----------
        None
        """
        self.roll(roll=roll)
        self.plot_elements(plotter=self.plotter, verbose=self.verbose)
    
    def generate_report(self) -> None:
        hwa_to_rack = 1 / 360 * 3.50 * 0.0254
        self.steer_sweep = np.linspace(-100, 100, 100) * hwa_to_rack
        self.heave_sweep = np.linspace(-2 * 0.0254, 2 * 0.0254, 4)

        # Store double wishbones
        self.FL_dw = self.full_suspension.Fr_axle.left
        self.FR_dw = self.full_suspension.Fr_axle.right

        ackermann_dict = {
            "Steer Steer": {"FL": [], "FR": []},
        }
        
        fig_1, ax_1 = plt.subplots(nrows=2, ncols=2)
        fig_1.set_size_inches(w=11, h=8.5)
        fig_1.suptitle("Steering Characteristics")
        fig_1.subplots_adjust(wspace=0.2, hspace=0.4)

        ax_1[0, 0].set_xlabel("Steer (in)")
        ax_1[0, 0].set_ylabel("Tire Heading (deg)")
        ax_1[0, 0].set_title("Tire Heading vs Steer")
        ax_1[0, 0].grid()

        ax_1[0, 1].set_xlabel("Steer (in)")
        ax_1[0, 1].set_ylabel("Tire Heading (deg)")
        ax_1[0, 1].set_title("Tire Heading vs Steer")
        ax_1[0, 1].grid()

        ax_1[1, 0].set_xlabel("Steer (in)")
        ax_1[1, 0].set_ylabel("Tire Heading (deg)")
        ax_1[1, 0].set_title("Tire Heading vs Steer")
        ax_1[1, 0].grid()

        ax_1[1, 1].set_xlabel("Steer (in)")
        ax_1[1, 1].set_ylabel("Ackermann Percent (%)")
        ax_1[1, 1].set_title("Ackermann Percent vs Steer")
        ax_1[1, 1].grid()

        ackermann_dict["Steer Steer"]["FL"] = []
        ackermann_dict["Steer Steer"]["FR"] = []

        for heave in [0.0001]:
            for steer in self.steer_sweep:
                self.full_suspension.hard_reset()
                
                self.steer(rack_displacement=steer)
                self.heave(heave=heave)

                ackermann_dict["Steer Steer"]["FL"].append(float(self.FL_dw.toe * 180 / np.pi))
                ackermann_dict["Steer Steer"]["FR"].append(float(self.FR_dw.toe * 180 / np.pi))
        
            ax_1[0, 0].plot(self.steer_sweep / hwa_to_rack, ackermann_dict["Steer Steer"]["FL"])
            ax_1[0, 1].plot(self.steer_sweep / hwa_to_rack, ackermann_dict["Steer Steer"]["FR"])

            ax_1[1, 0].plot(self.steer_sweep / hwa_to_rack, ackermann_dict["Steer Steer"]["FL"], label="Left")
            ax_1[1, 0].plot(self.steer_sweep / hwa_to_rack, ackermann_dict["Steer Steer"]["FR"], label="Right")

            ackermann_lst = []
            for i in range(len(self.steer_sweep)):
                if self.steer_sweep[i] <= 0:
                    ackermann_lst.append((ackermann_dict["Steer Steer"]["FR"][i] - ackermann_dict["Steer Steer"]["FL"][i]) / ackermann_dict["Steer Steer"]["FR"][i] * 100)
                else:
                    ackermann_lst.append((ackermann_dict["Steer Steer"]["FL"][i] - ackermann_dict["Steer Steer"]["FR"][i]) / ackermann_dict["Steer Steer"]["FL"][i] * 100)

            ax_1[1, 1].plot(self.steer_sweep / hwa_to_rack, ackermann_lst)

        ax_1[1, 0].legend()

        ax_1[1, 1].set_ylim([min(ackermann_lst) * 5, max(ackermann_lst) * 5])

        # plt.show()
    
    def generate_kin_plots(self, steer_sweep: np.ndarray, heave_sweep: np.ndarray, pitch_sweep: np.ndarray, roll_sweep: np.ndarray) -> None:
        self.steer_sweep = steer_sweep
        self.heave_sweep = heave_sweep
        self.pitch_sweep = pitch_sweep
        self.roll_sweep = roll_sweep

        # Store double wishbones
        self.FL_dw = self.full_suspension.Fr_axle.left
        self.FR_dw = self.full_suspension.Fr_axle.right
        self.RL_dw = self.full_suspension.Rr_axle.left
        self.RR_dw = self.full_suspension.Rr_axle.right

        kin_dict = {
            "Bump Camber Full": {"FL": [], "FR": [], "RL": [], "RR": []}, 
            "Bump Toe Full": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump Caster Full": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump FVIC Migration Full": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump SVIC Migration Full": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump FAPz Migration Full": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump CPx Migration Full": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump CPy Migration Full": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump Kw Full": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump Kr Full": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump Kp Full": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll Camber Full": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll Toe Full": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll Caster Full": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll FVIC Migration Full": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll SVIC Migration Full": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll FAPz Migration Full": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll KinRC z Migration Full": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll KinRC y Migration Full": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll CPx Migration Full": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll CPy Migration Full": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll Kw Full": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll Kr Full": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll Kp Full": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump Camber Lite": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump Toe Lite": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump Caster Lite": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump FVIC Migration Lite": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump SVIC Migration Lite": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump FAPz Migration Lite": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump CPx Migration Lite": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump CPy Migration Lite": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump Kw Lite": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump Kr Lite": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump Kp Lite": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll Camber Lite": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll Toe Lite": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll Caster Lite": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll FVIC Migration Lite": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll SVIC Migration Lite": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll FAPz Migration Lite": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll CPx Migration Lite": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll CPy Migration Lite": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll Kw Lite": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll Kr Lite": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll Kp Lite": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump Camber Cached": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump Toe Cached": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump Caster Cached": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump FVIC Migration Cached": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump SVIC Migration Cached": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump FAPz Migration Cached": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump CPx Migration Cached": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump CPy Migration Cached": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump Kw Cached": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump Kr Cached": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Bump Kp Cached": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll Camber Cached": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll Toe Cached": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll Caster Cached": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll FVIC Migration Cached": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll SVIC Migration Cached": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll FAPz Migration Cached": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll CPx Migration Cached": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll CPy Migration Cached": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll Scrub Full": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll Kw Cached": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll Kr Cached": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Roll Kp Cached": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Steer Camber Full_-2": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Steer Camber Full_-1": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Steer Camber Full": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Steer Camber Full_1": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Steer Camber Full_2": {"FL": [], "FR": [], "RL": [], "RR": []}, 
            "Steer Toe Full": {"FL": [], "FR": [], "RL": [], "RR": []},
            "Steer Caster Full": {"FL": [], "FR": [], "RL": [], "RR": []},
        }
        
        # Simplified Kinematics
        for heave in self.heave_sweep:
            self.full_suspension.hard_reset()
            self.generate_kin_helper(steer=0, heave=heave, pitch=0, roll=0)
            kin_dict["Bump Camber Lite"]["FL"].append(-self.FL_dw.inclination_angle * 180 / np.pi)
            kin_dict["Bump Camber Lite"]["FR"].append(self.FR_dw.inclination_angle * 180 / np.pi)
            kin_dict["Bump Camber Lite"]["RL"].append(-self.RL_dw.inclination_angle * 180 / np.pi)
            kin_dict["Bump Camber Lite"]["RR"].append(self.RR_dw.inclination_angle * 180 / np.pi)

            kin_dict["Bump Toe Lite"]["FL"].append(-self.FL_dw.toe * 180 / np.pi)
            kin_dict["Bump Toe Lite"]["FR"].append(self.FR_dw.toe * 180 / np.pi)
            kin_dict["Bump Toe Lite"]["RL"].append(-self.RL_dw.toe * 180 / np.pi)
            kin_dict["Bump Toe Lite"]["RR"].append(self.RR_dw.toe * 180 / np.pi)

            kin_dict["Bump Caster Lite"]["FL"].append(self.FL_dw.caster * 180 / np.pi)
            kin_dict["Bump Caster Lite"]["FR"].append(self.FR_dw.caster * 180 / np.pi)
            kin_dict["Bump Caster Lite"]["RL"].append(self.RL_dw.caster * 180 / np.pi)
            kin_dict["Bump Caster Lite"]["RR"].append(self.RR_dw.caster * 180 / np.pi)

            kin_dict["Bump FVIC Migration Lite"]["FL"].append(self.FL_dw.FVIC_position[2])
            kin_dict["Bump FVIC Migration Lite"]["FR"].append(self.FR_dw.FVIC_position[2])
            kin_dict["Bump FVIC Migration Lite"]["RL"].append(self.RL_dw.FVIC_position[2])
            kin_dict["Bump FVIC Migration Lite"]["RR"].append(self.RR_dw.FVIC_position[2])

            kin_dict["Bump SVIC Migration Lite"]["FL"].append(self.FL_dw.SVIC_position[2])
            kin_dict["Bump SVIC Migration Lite"]["FR"].append(self.FR_dw.SVIC_position[2])
            kin_dict["Bump SVIC Migration Lite"]["RL"].append(self.RL_dw.SVIC_position[2])
            kin_dict["Bump SVIC Migration Lite"]["RR"].append(self.RR_dw.SVIC_position[2])

            kin_dict["Bump FAPz Migration Lite"]["FL"].append(self.FL_dw.FV_FAP_position[2])
            kin_dict["Bump FAPz Migration Lite"]["FR"].append(self.FR_dw.FV_FAP_position[2])
            kin_dict["Bump FAPz Migration Lite"]["RL"].append(self.RL_dw.FV_FAP_position[2])
            kin_dict["Bump FAPz Migration Lite"]["RR"].append(self.RR_dw.FV_FAP_position[2])

            kin_dict["Bump CPx Migration Lite"]["FL"].append(self.FL_dw.contact_patch.position[0])
            kin_dict["Bump CPx Migration Lite"]["FR"].append(self.FR_dw.contact_patch.position[0])
            kin_dict["Bump CPx Migration Lite"]["RL"].append(self.RL_dw.contact_patch.position[0])
            kin_dict["Bump CPx Migration Lite"]["RR"].append(self.RR_dw.contact_patch.position[0])

            kin_dict["Bump CPy Migration Lite"]["FL"].append(self.FL_dw.contact_patch.position[1])
            kin_dict["Bump CPy Migration Lite"]["FR"].append(self.FR_dw.contact_patch.position[1])
            kin_dict["Bump CPy Migration Lite"]["RL"].append(self.RL_dw.contact_patch.position[1])
            kin_dict["Bump CPy Migration Lite"]["RR"].append(self.RR_dw.contact_patch.position[1])

            kin_dict["Bump Kw Lite"]["FL"].append(self.FL_dw.wheelrate)
            kin_dict["Bump Kw Lite"]["FR"].append(self.FR_dw.wheelrate)
            kin_dict["Bump Kw Lite"]["RL"].append(self.RL_dw.wheelrate)
            kin_dict["Bump Kw Lite"]["RR"].append(self.RR_dw.wheelrate)

            kin_dict["Bump Kr Lite"]["FL"].append(self.Fr_axle.roll_stiffness)
            kin_dict["Bump Kr Lite"]["FR"].append(self.Fr_axle.roll_stiffness)
            kin_dict["Bump Kr Lite"]["RL"].append(self.Rr_axle.roll_stiffness)
            kin_dict["Bump Kr Lite"]["RR"].append(self.Rr_axle.roll_stiffness)

            kin_dict["Bump Kp Lite"]["FL"].append(self.full_suspension.pitch_stiffness)
            kin_dict["Bump Kp Lite"]["FR"].append(self.full_suspension.pitch_stiffness)
            kin_dict["Bump Kp Lite"]["RL"].append(self.full_suspension.pitch_stiffness)
            kin_dict["Bump Kp Lite"]["RR"].append(self.full_suspension.pitch_stiffness)
        
        for roll in self.roll_sweep:
            self.full_suspension.hard_reset()
            self.generate_kin_helper(steer=0, heave=0, pitch=0, roll=roll)
            kin_dict["Roll Camber Lite"]["FL"].append(-self.FL_dw.inclination_angle * 180 / np.pi - roll * 180 / np.pi)
            kin_dict["Roll Camber Lite"]["FR"].append(self.FR_dw.inclination_angle * 180 / np.pi + roll * 180 / np.pi)
            kin_dict["Roll Camber Lite"]["RL"].append(-self.RL_dw.inclination_angle * 180 / np.pi - roll * 180 / np.pi)
            kin_dict["Roll Camber Lite"]["RR"].append(self.RR_dw.inclination_angle * 180 / np.pi + roll * 180 / np.pi)

            kin_dict["Roll Toe Lite"]["FL"].append(-self.FL_dw.toe * 180 / np.pi)
            kin_dict["Roll Toe Lite"]["FR"].append(self.FR_dw.toe * 180 / np.pi)
            kin_dict["Roll Toe Lite"]["RL"].append(-self.RL_dw.toe * 180 / np.pi)
            kin_dict["Roll Toe Lite"]["RR"].append(self.RR_dw.toe * 180 / np.pi)

            kin_dict["Roll Caster Lite"]["FL"].append(self.FL_dw.caster * 180 / np.pi)
            kin_dict["Roll Caster Lite"]["FR"].append(self.FR_dw.caster * 180 / np.pi)
            kin_dict["Roll Caster Lite"]["RL"].append(self.RL_dw.caster * 180 / np.pi)
            kin_dict["Roll Caster Lite"]["RR"].append(self.RR_dw.caster * 180 / np.pi)

            kin_dict["Roll FVIC Migration Lite"]["FL"].append(self.FL_dw.FVIC_position[2])
            kin_dict["Roll FVIC Migration Lite"]["FR"].append(self.FR_dw.FVIC_position[2])
            kin_dict["Roll FVIC Migration Lite"]["RL"].append(self.RL_dw.FVIC_position[2])
            kin_dict["Roll FVIC Migration Lite"]["RR"].append(self.RR_dw.FVIC_position[2])

            kin_dict["Roll SVIC Migration Lite"]["FL"].append(self.FL_dw.SVIC_position[2])
            kin_dict["Roll SVIC Migration Lite"]["FR"].append(self.FR_dw.SVIC_position[2])
            kin_dict["Roll SVIC Migration Lite"]["RL"].append(self.RL_dw.SVIC_position[2])
            kin_dict["Roll SVIC Migration Lite"]["RR"].append(self.RR_dw.SVIC_position[2])

            kin_dict["Roll FAPz Migration Lite"]["FL"].append(self.FL_dw.FV_FAP_position[2])
            kin_dict["Roll FAPz Migration Lite"]["FR"].append(self.FR_dw.FV_FAP_position[2])
            kin_dict["Roll FAPz Migration Lite"]["RL"].append(self.RL_dw.FV_FAP_position[2])
            kin_dict["Roll FAPz Migration Lite"]["RR"].append(self.RR_dw.FV_FAP_position[2])

            kin_dict["Roll CPx Migration Lite"]["FL"].append(self.FL_dw.contact_patch.position[0])
            kin_dict["Roll CPx Migration Lite"]["FR"].append(self.FR_dw.contact_patch.position[0])
            kin_dict["Roll CPx Migration Lite"]["RL"].append(self.RL_dw.contact_patch.position[0])
            kin_dict["Roll CPx Migration Lite"]["RR"].append(self.RR_dw.contact_patch.position[0])

            kin_dict["Roll CPy Migration Lite"]["FL"].append(self.FL_dw.contact_patch.position[1])
            kin_dict["Roll CPy Migration Lite"]["FR"].append(self.FR_dw.contact_patch.position[1])
            kin_dict["Roll CPy Migration Lite"]["RL"].append(self.RL_dw.contact_patch.position[1])
            kin_dict["Roll CPy Migration Lite"]["RR"].append(self.RR_dw.contact_patch.position[1])

            kin_dict["Roll Kw Lite"]["FL"].append(self.FL_dw.wheelrate)
            kin_dict["Roll Kw Lite"]["FR"].append(self.FR_dw.wheelrate)
            kin_dict["Roll Kw Lite"]["RL"].append(self.RL_dw.wheelrate)
            kin_dict["Roll Kw Lite"]["RR"].append(self.RR_dw.wheelrate)

            kin_dict["Roll Kr Lite"]["FL"].append(self.Fr_axle.roll_stiffness)
            kin_dict["Roll Kr Lite"]["FR"].append(self.Fr_axle.roll_stiffness)
            kin_dict["Roll Kr Lite"]["RL"].append(self.Rr_axle.roll_stiffness)
            kin_dict["Roll Kr Lite"]["RR"].append(self.Rr_axle.roll_stiffness)

            kin_dict["Roll Kp Lite"]["FL"].append(self.full_suspension.pitch_stiffness)
            kin_dict["Roll Kp Lite"]["FR"].append(self.full_suspension.pitch_stiffness)
            kin_dict["Roll Kp Lite"]["RL"].append(self.full_suspension.pitch_stiffness)
            kin_dict["Roll Kp Lite"]["RR"].append(self.full_suspension.pitch_stiffness)

        # Full Kinematics
        # self.full_suspension.hard_reset()
        for heave in self.heave_sweep:
            self.full_suspension.hard_reset()
            self.heave(heave=heave)
            kin_dict["Bump Camber Full"]["FL"].append(-self.FL_dw.inclination_angle * 180 / np.pi)
            kin_dict["Bump Camber Full"]["FR"].append(self.FR_dw.inclination_angle * 180 / np.pi)
            kin_dict["Bump Camber Full"]["RL"].append(-self.RL_dw.inclination_angle * 180 / np.pi)
            kin_dict["Bump Camber Full"]["RR"].append(self.RR_dw.inclination_angle * 180 / np.pi)

            kin_dict["Bump Toe Full"]["FL"].append(-self.FL_dw.toe * 180 / np.pi)
            kin_dict["Bump Toe Full"]["FR"].append(self.FR_dw.toe * 180 / np.pi)
            kin_dict["Bump Toe Full"]["RL"].append(-self.RL_dw.toe * 180 / np.pi)
            kin_dict["Bump Toe Full"]["RR"].append(self.RR_dw.toe * 180 / np.pi)

            kin_dict["Bump Caster Full"]["FL"].append(self.FL_dw.caster * 180 / np.pi)
            kin_dict["Bump Caster Full"]["FR"].append(self.FR_dw.caster * 180 / np.pi)
            kin_dict["Bump Caster Full"]["RL"].append(self.RL_dw.caster * 180 / np.pi)
            kin_dict["Bump Caster Full"]["RR"].append(self.RR_dw.caster * 180 / np.pi)

            kin_dict["Bump FVIC Migration Full"]["FL"].append(self.FL_dw.FVIC_position[2])
            kin_dict["Bump FVIC Migration Full"]["FR"].append(self.FR_dw.FVIC_position[2])
            kin_dict["Bump FVIC Migration Full"]["RL"].append(self.RL_dw.FVIC_position[2])
            kin_dict["Bump FVIC Migration Full"]["RR"].append(self.RR_dw.FVIC_position[2])

            kin_dict["Bump SVIC Migration Full"]["FL"].append(self.FL_dw.SVIC_position[2])
            kin_dict["Bump SVIC Migration Full"]["FR"].append(self.FR_dw.SVIC_position[2])
            kin_dict["Bump SVIC Migration Full"]["RL"].append(self.RL_dw.SVIC_position[2])
            kin_dict["Bump SVIC Migration Full"]["RR"].append(self.RR_dw.SVIC_position[2])

            kin_dict["Bump FAPz Migration Full"]["FL"].append(self.FL_dw.FV_FAP_position[2])
            kin_dict["Bump FAPz Migration Full"]["FR"].append(self.FR_dw.FV_FAP_position[2])
            kin_dict["Bump FAPz Migration Full"]["RL"].append(self.RL_dw.FV_FAP_position[2])
            kin_dict["Bump FAPz Migration Full"]["RR"].append(self.RR_dw.FV_FAP_position[2])

            kin_dict["Bump CPx Migration Full"]["FL"].append(self.FL_dw.contact_patch.position[0])
            kin_dict["Bump CPx Migration Full"]["FR"].append(self.FR_dw.contact_patch.position[0])
            kin_dict["Bump CPx Migration Full"]["RL"].append(self.RL_dw.contact_patch.position[0])
            kin_dict["Bump CPx Migration Full"]["RR"].append(self.RR_dw.contact_patch.position[0])

            kin_dict["Bump CPy Migration Full"]["FL"].append(self.FL_dw.contact_patch.position[1])
            kin_dict["Bump CPy Migration Full"]["FR"].append(self.FR_dw.contact_patch.position[1])
            kin_dict["Bump CPy Migration Full"]["RL"].append(self.RL_dw.contact_patch.position[1])
            kin_dict["Bump CPy Migration Full"]["RR"].append(self.RR_dw.contact_patch.position[1])

            kin_dict["Bump Kw Full"]["FL"].append(self.FL_dw.wheelrate)
            kin_dict["Bump Kw Full"]["FR"].append(self.FR_dw.wheelrate)
            kin_dict["Bump Kw Full"]["RL"].append(self.RL_dw.wheelrate)
            kin_dict["Bump Kw Full"]["RR"].append(self.RR_dw.wheelrate)

            kin_dict["Bump Kr Full"]["FL"].append(self.Fr_axle.roll_stiffness)
            kin_dict["Bump Kr Full"]["FR"].append(self.Fr_axle.roll_stiffness)
            kin_dict["Bump Kr Full"]["RL"].append(self.Rr_axle.roll_stiffness)
            kin_dict["Bump Kr Full"]["RR"].append(self.Rr_axle.roll_stiffness)

            kin_dict["Bump Kp Full"]["FL"].append(self.full_suspension.pitch_stiffness)
            kin_dict["Bump Kp Full"]["FR"].append(self.full_suspension.pitch_stiffness)
            kin_dict["Bump Kp Full"]["RL"].append(self.full_suspension.pitch_stiffness)
            kin_dict["Bump Kp Full"]["RR"].append(self.full_suspension.pitch_stiffness)
        
        # self.full_suspension.hard_reset()
        for roll in self.roll_sweep:
            self.full_suspension.hard_reset()
            self.roll(roll=roll * 180 / np.pi)
            kin_dict["Roll Camber Full"]["FL"].append(-self.FL_dw.inclination_angle * 180 / np.pi)
            kin_dict["Roll Camber Full"]["FR"].append(self.FR_dw.inclination_angle * 180 / np.pi)
            kin_dict["Roll Camber Full"]["RL"].append(-self.RL_dw.inclination_angle * 180 / np.pi)
            kin_dict["Roll Camber Full"]["RR"].append(self.RR_dw.inclination_angle * 180 / np.pi)

            kin_dict["Roll Toe Full"]["FL"].append(-self.FL_dw.toe * 180 / np.pi)
            kin_dict["Roll Toe Full"]["FR"].append(self.FR_dw.toe * 180 / np.pi)
            kin_dict["Roll Toe Full"]["RL"].append(-self.RL_dw.toe * 180 / np.pi)
            kin_dict["Roll Toe Full"]["RR"].append(self.RR_dw.toe * 180 / np.pi)

            kin_dict["Roll Caster Full"]["FL"].append(self.FL_dw.caster * 180 / np.pi)
            kin_dict["Roll Caster Full"]["FR"].append(self.FR_dw.caster * 180 / np.pi)
            kin_dict["Roll Caster Full"]["RL"].append(self.RL_dw.caster * 180 / np.pi)
            kin_dict["Roll Caster Full"]["RR"].append(self.RR_dw.caster * 180 / np.pi)

            kin_dict["Roll FVIC Migration Full"]["FL"].append(self.FL_dw.FVIC_position[2])
            kin_dict["Roll FVIC Migration Full"]["FR"].append(self.FR_dw.FVIC_position[2])
            kin_dict["Roll FVIC Migration Full"]["RL"].append(self.RL_dw.FVIC_position[2])
            kin_dict["Roll FVIC Migration Full"]["RR"].append(self.RR_dw.FVIC_position[2])

            kin_dict["Roll KinRC z Migration Full"]["FL"].append(self.Fr_axle.kin_RC.vertical_position)
            kin_dict["Roll KinRC z Migration Full"]["FR"].append(self.Fr_axle.kin_RC.vertical_position)
            kin_dict["Roll KinRC z Migration Full"]["RL"].append(self.Rr_axle.kin_RC.vertical_position)
            kin_dict["Roll KinRC z Migration Full"]["RR"].append(self.Rr_axle.kin_RC.vertical_position)

            kin_dict["Roll KinRC y Migration Full"]["FL"].append(self.Fr_axle.kin_RC.lateral_position)
            kin_dict["Roll KinRC y Migration Full"]["FR"].append(self.Fr_axle.kin_RC.lateral_position)
            kin_dict["Roll KinRC y Migration Full"]["RL"].append(self.Rr_axle.kin_RC.lateral_position)
            kin_dict["Roll KinRC y Migration Full"]["RR"].append(self.Rr_axle.kin_RC.lateral_position)

            kin_dict["Roll Scrub Full"]["FL"].append(self.FL_dw.scrub)
            kin_dict["Roll Scrub Full"]["FR"].append(self.FR_dw.scrub)
            kin_dict["Roll Scrub Full"]["RL"].append(self.RL_dw.scrub)
            kin_dict["Roll Scrub Full"]["RR"].append(self.RR_dw.scrub)

            kin_dict["Roll SVIC Migration Full"]["FL"].append(self.FL_dw.SVIC_position[2])
            kin_dict["Roll SVIC Migration Full"]["FR"].append(self.FR_dw.SVIC_position[2])
            kin_dict["Roll SVIC Migration Full"]["RL"].append(self.RL_dw.SVIC_position[2])
            kin_dict["Roll SVIC Migration Full"]["RR"].append(self.RR_dw.SVIC_position[2])

            kin_dict["Roll FAPz Migration Full"]["FL"].append(self.FL_dw.FV_FAP_position[2])
            kin_dict["Roll FAPz Migration Full"]["FR"].append(self.FR_dw.FV_FAP_position[2])
            kin_dict["Roll FAPz Migration Full"]["RL"].append(self.RL_dw.FV_FAP_position[2])
            kin_dict["Roll FAPz Migration Full"]["RR"].append(self.RR_dw.FV_FAP_position[2])

            kin_dict["Roll CPx Migration Full"]["FL"].append(self.FL_dw.contact_patch.position[0])
            kin_dict["Roll CPx Migration Full"]["FR"].append(self.FR_dw.contact_patch.position[0])
            kin_dict["Roll CPx Migration Full"]["RL"].append(self.RL_dw.contact_patch.position[0])
            kin_dict["Roll CPx Migration Full"]["RR"].append(self.RR_dw.contact_patch.position[0])

            kin_dict["Roll CPy Migration Full"]["FL"].append(self.FL_dw.contact_patch.position[1])
            kin_dict["Roll CPy Migration Full"]["FR"].append(self.FR_dw.contact_patch.position[1])
            kin_dict["Roll CPy Migration Full"]["RL"].append(self.RL_dw.contact_patch.position[1])
            kin_dict["Roll CPy Migration Full"]["RR"].append(self.RR_dw.contact_patch.position[1])

            kin_dict["Roll Kw Full"]["FL"].append(self.FL_dw.wheelrate)
            kin_dict["Roll Kw Full"]["FR"].append(self.FR_dw.wheelrate)
            kin_dict["Roll Kw Full"]["RL"].append(self.RL_dw.wheelrate)
            kin_dict["Roll Kw Full"]["RR"].append(self.RR_dw.wheelrate)

            kin_dict["Roll Kr Full"]["FL"].append(self.Fr_axle.roll_stiffness)
            kin_dict["Roll Kr Full"]["FR"].append(self.Fr_axle.roll_stiffness)
            kin_dict["Roll Kr Full"]["RL"].append(self.Rr_axle.roll_stiffness)
            kin_dict["Roll Kr Full"]["RR"].append(self.Rr_axle.roll_stiffness)

            kin_dict["Roll Kp Full"]["FL"].append(self.full_suspension.pitch_stiffness)
            kin_dict["Roll Kp Full"]["FR"].append(self.full_suspension.pitch_stiffness)
            kin_dict["Roll Kp Full"]["RL"].append(self.full_suspension.pitch_stiffness)
            kin_dict["Roll Kp Full"]["RR"].append(self.full_suspension.pitch_stiffness)

        for steer in self.steer_sweep:
            for roll in [-2, -1, 0.00001, 1, 2]:
                self.full_suspension.hard_reset()
                self.steer(rack_displacement=steer)
                self.roll(roll=roll)

                if int(roll) == roll:
                    kin_dict[f"Steer Camber Full_{roll}"]["FL"].append(-self.FL_dw.inclination_angle * 180 / np.pi)
                    kin_dict[f"Steer Camber Full_{roll}"]["FR"].append(self.FR_dw.inclination_angle * 180 / np.pi)
                    kin_dict[f"Steer Camber Full_{roll}"]["RL"].append(-self.RL_dw.inclination_angle * 180 / np.pi)
                    kin_dict[f"Steer Camber Full_{roll}"]["RR"].append(self.RR_dw.inclination_angle * 180 / np.pi)
                else:
                    kin_dict[f"Steer Camber Full"]["FL"].append(-self.FL_dw.inclination_angle * 180 / np.pi)
                    kin_dict[f"Steer Camber Full"]["FR"].append(self.FR_dw.inclination_angle * 180 / np.pi)
                    kin_dict[f"Steer Camber Full"]["RL"].append(-self.RL_dw.inclination_angle * 180 / np.pi)
                    kin_dict[f"Steer Camber Full"]["RR"].append(self.RR_dw.inclination_angle * 180 / np.pi)
                
            self.full_suspension.hard_reset()
            self.steer(rack_displacement=steer)
            kin_dict["Steer Toe Full"]["FL"].append(-self.FL_dw.toe * 180 / np.pi)
            kin_dict["Steer Toe Full"]["FR"].append(self.FR_dw.toe * 180 / np.pi)
            kin_dict["Steer Toe Full"]["RL"].append(-self.RL_dw.toe * 180 / np.pi)
            kin_dict["Steer Toe Full"]["RR"].append(self.RR_dw.toe * 180 / np.pi)

            kin_dict["Steer Caster Full"]["FL"].append(self.FL_dw.caster * 180 / np.pi)
            kin_dict["Steer Caster Full"]["FR"].append(self.FR_dw.caster * 180 / np.pi)
            kin_dict["Steer Caster Full"]["RL"].append(self.RL_dw.caster * 180 / np.pi)
            kin_dict["Steer Caster Full"]["RR"].append(self.RR_dw.caster * 180 / np.pi)

        self.full_suspension.hard_reset()

        # Cached Kinematics
        for heave in self.heave_sweep:
            kin_dict["Bump Camber Cached"]["FL"].append(-self.FL_gamma_lookup(x=0, y=heave, z=0, w=0) * 180 / np.pi)
            kin_dict["Bump Camber Cached"]["FR"].append(self.FR_gamma_lookup(x=0, y=heave, z=0, w=0) * 180 / np.pi)
            kin_dict["Bump Camber Cached"]["RL"].append(-self.RL_gamma_lookup(x=0, y=heave, z=0, w=0) * 180 / np.pi)
            kin_dict["Bump Camber Cached"]["RR"].append(self.RR_gamma_lookup(x=0, y=heave, z=0, w=0) * 180 / np.pi)

            kin_dict["Bump Toe Cached"]["FL"].append(-self.FL_toe_lookup(x=0, y=heave, z=0, w=0) * 180 / np.pi)
            kin_dict["Bump Toe Cached"]["FR"].append(self.FR_toe_lookup(x=0, y=heave, z=0, w=0) * 180 / np.pi)
            kin_dict["Bump Toe Cached"]["RL"].append(-self.RL_toe_lookup(x=0, y=heave, z=0, w=0) * 180 / np.pi)
            kin_dict["Bump Toe Cached"]["RR"].append(self.RR_toe_lookup(x=0, y=heave, z=0, w=0) * 180 / np.pi)

            kin_dict["Bump Caster Cached"]["FL"].append(self.FL_caster_lookup(x=0, y=heave, z=0, w=0) * 180 / np.pi)
            kin_dict["Bump Caster Cached"]["FR"].append(self.FR_caster_lookup(x=0, y=heave, z=0, w=0) * 180 / np.pi)
            kin_dict["Bump Caster Cached"]["RL"].append(self.RL_caster_lookup(x=0, y=heave, z=0, w=0) * 180 / np.pi)
            kin_dict["Bump Caster Cached"]["RR"].append(self.RR_caster_lookup(x=0, y=heave, z=0, w=0) * 180 / np.pi)

            kin_dict["Bump FVIC Migration Cached"]["FL"].append(self.FL_FV_ICz_lookup(x=0, y=heave, z=0, w=0))
            kin_dict["Bump FVIC Migration Cached"]["FR"].append(self.FR_FV_ICz_lookup(x=0, y=heave, z=0, w=0))
            kin_dict["Bump FVIC Migration Cached"]["RL"].append(self.RL_FV_ICz_lookup(x=0, y=heave, z=0, w=0))
            kin_dict["Bump FVIC Migration Cached"]["RR"].append(self.RR_FV_ICz_lookup(x=0, y=heave, z=0, w=0))

            kin_dict["Bump SVIC Migration Cached"]["FL"].append(self.FL_SV_ICz_lookup(x=0, y=heave, z=0, w=0))
            kin_dict["Bump SVIC Migration Cached"]["FR"].append(self.FR_SV_ICz_lookup(x=0, y=heave, z=0, w=0))
            kin_dict["Bump SVIC Migration Cached"]["RL"].append(self.RL_SV_ICz_lookup(x=0, y=heave, z=0, w=0))
            kin_dict["Bump SVIC Migration Cached"]["RR"].append(self.RR_SV_ICz_lookup(x=0, y=heave, z=0, w=0))

            kin_dict["Bump FAPz Migration Cached"]["FL"].append(self.FL_FV_FAPz_lookup(x=0, y=heave, z=0, w=0))
            kin_dict["Bump FAPz Migration Cached"]["FR"].append(self.FR_FV_FAPz_lookup(x=0, y=heave, z=0, w=0))
            kin_dict["Bump FAPz Migration Cached"]["RL"].append(self.RL_FV_FAPz_lookup(x=0, y=heave, z=0, w=0))
            kin_dict["Bump FAPz Migration Cached"]["RR"].append(self.RR_FV_FAPz_lookup(x=0, y=heave, z=0, w=0))

            kin_dict["Bump CPx Migration Cached"]["FL"].append(self.FL_cpx_lookup(x=0, y=heave, z=0, w=0))
            kin_dict["Bump CPx Migration Cached"]["FR"].append(self.FR_cpx_lookup(x=0, y=heave, z=0, w=0))
            kin_dict["Bump CPx Migration Cached"]["RL"].append(self.RL_cpx_lookup(x=0, y=heave, z=0, w=0))
            kin_dict["Bump CPx Migration Cached"]["RR"].append(self.RR_cpx_lookup(x=0, y=heave, z=0, w=0))

            kin_dict["Bump CPy Migration Cached"]["FL"].append(self.FL_cpy_lookup(x=0, y=heave, z=0, w=0))
            kin_dict["Bump CPy Migration Cached"]["FR"].append(self.FR_cpy_lookup(x=0, y=heave, z=0, w=0))
            kin_dict["Bump CPy Migration Cached"]["RL"].append(self.RL_cpy_lookup(x=0, y=heave, z=0, w=0))
            kin_dict["Bump CPy Migration Cached"]["RR"].append(self.RR_cpy_lookup(x=0, y=heave, z=0, w=0))

            kin_dict["Bump Kw Cached"]["FL"].append(self.FL_Kw_lookup(x=0, y=heave, z=0, w=0))
            kin_dict["Bump Kw Cached"]["FR"].append(self.FR_Kw_lookup(x=0, y=heave, z=0, w=0))
            kin_dict["Bump Kw Cached"]["RL"].append(self.RL_Kw_lookup(x=0, y=heave, z=0, w=0))
            kin_dict["Bump Kw Cached"]["RR"].append(self.RR_Kw_lookup(x=0, y=heave, z=0, w=0))

            kin_dict["Bump Kr Cached"]["FL"].append(self.Fr_Kr_lookup(x=0, y=heave, z=0, w=0))
            kin_dict["Bump Kr Cached"]["FR"].append(self.Fr_Kr_lookup(x=0, y=heave, z=0, w=0))
            kin_dict["Bump Kr Cached"]["RL"].append(self.Rr_Kr_lookup(x=0, y=heave, z=0, w=0))
            kin_dict["Bump Kr Cached"]["RR"].append(self.Rr_Kr_lookup(x=0, y=heave, z=0, w=0))

            kin_dict["Bump Kp Cached"]["FL"].append(self.Kp_lookup(x=0, y=heave, z=0, w=0))
            kin_dict["Bump Kp Cached"]["FR"].append(self.Kp_lookup(x=0, y=heave, z=0, w=0))
            kin_dict["Bump Kp Cached"]["RL"].append(self.Kp_lookup(x=0, y=heave, z=0, w=0))
            kin_dict["Bump Kp Cached"]["RR"].append(self.Kp_lookup(x=0, y=heave, z=0, w=0))

        for roll in self.roll_sweep:
            kin_dict["Roll Camber Cached"]["FL"].append(-self.FL_gamma_lookup(x=0, y=0, z=0, w=roll) * 180 / np.pi)
            kin_dict["Roll Camber Cached"]["FR"].append(self.FR_gamma_lookup(x=0, y=0, z=0, w=roll) * 180 / np.pi)
            kin_dict["Roll Camber Cached"]["RL"].append(-self.RL_gamma_lookup(x=0, y=0, z=0, w=roll) * 180 / np.pi)
            kin_dict["Roll Camber Cached"]["RR"].append(self.RR_gamma_lookup(x=0, y=0, z=0, w=roll) * 180 / np.pi)

            kin_dict["Roll Toe Cached"]["FL"].append(-self.FL_toe_lookup(x=0, y=0, z=0, w=roll) * 180 / np.pi)
            kin_dict["Roll Toe Cached"]["FR"].append(self.FR_toe_lookup(x=0, y=0, z=0, w=roll) * 180 / np.pi)
            kin_dict["Roll Toe Cached"]["RL"].append(-self.RL_toe_lookup(x=0, y=0, z=0, w=roll) * 180 / np.pi)
            kin_dict["Roll Toe Cached"]["RR"].append(self.RR_toe_lookup(x=0, y=0, z=0, w=roll) * 180 / np.pi)

            kin_dict["Roll Caster Cached"]["FL"].append(self.FL_caster_lookup(x=0, y=0, z=0, w=roll) * 180 / np.pi)
            kin_dict["Roll Caster Cached"]["FR"].append(self.FR_caster_lookup(x=0, y=0, z=0, w=roll) * 180 / np.pi)
            kin_dict["Roll Caster Cached"]["RL"].append(self.RL_caster_lookup(x=0, y=0, z=0, w=roll) * 180 / np.pi)
            kin_dict["Roll Caster Cached"]["RR"].append(self.RR_caster_lookup(x=0, y=0, z=0, w=roll) * 180 / np.pi)

            kin_dict["Roll FVIC Migration Cached"]["FL"].append(self.FL_FV_ICz_lookup(x=0, y=0, z=0, w=roll))
            kin_dict["Roll FVIC Migration Cached"]["FR"].append(self.FR_FV_ICz_lookup(x=0, y=0, z=0, w=roll))
            kin_dict["Roll FVIC Migration Cached"]["RL"].append(self.RL_FV_ICz_lookup(x=0, y=0, z=0, w=roll))
            kin_dict["Roll FVIC Migration Cached"]["RR"].append(self.RR_FV_ICz_lookup(x=0, y=0, z=0, w=roll))

            kin_dict["Roll SVIC Migration Cached"]["FL"].append(self.FL_SV_ICz_lookup(x=0, y=0, z=0, w=roll))
            kin_dict["Roll SVIC Migration Cached"]["FR"].append(self.FR_SV_ICz_lookup(x=0, y=0, z=0, w=roll))
            kin_dict["Roll SVIC Migration Cached"]["RL"].append(self.RL_SV_ICz_lookup(x=0, y=0, z=0, w=roll))
            kin_dict["Roll SVIC Migration Cached"]["RR"].append(self.RR_SV_ICz_lookup(x=0, y=0, z=0, w=roll))

            kin_dict["Roll FAPz Migration Cached"]["FL"].append(self.FL_FV_FAPz_lookup(x=0, y=0, z=0, w=roll))
            kin_dict["Roll FAPz Migration Cached"]["FR"].append(self.FR_FV_FAPz_lookup(x=0, y=0, z=0, w=roll))
            kin_dict["Roll FAPz Migration Cached"]["RL"].append(self.RL_FV_FAPz_lookup(x=0, y=0, z=0, w=roll))
            kin_dict["Roll FAPz Migration Cached"]["RR"].append(self.RR_FV_FAPz_lookup(x=0, y=0, z=0, w=roll))

            kin_dict["Roll CPx Migration Cached"]["FL"].append(self.FL_cpx_lookup(x=0, y=0, z=0, w=roll))
            kin_dict["Roll CPx Migration Cached"]["FR"].append(self.FR_cpx_lookup(x=0, y=0, z=0, w=roll))
            kin_dict["Roll CPx Migration Cached"]["RL"].append(self.RL_cpx_lookup(x=0, y=0, z=0, w=roll))
            kin_dict["Roll CPx Migration Cached"]["RR"].append(self.RR_cpx_lookup(x=0, y=0, z=0, w=roll))

            kin_dict["Roll CPy Migration Cached"]["FL"].append(self.FL_cpy_lookup(x=0, y=0, z=0, w=roll))
            kin_dict["Roll CPy Migration Cached"]["FR"].append(self.FR_cpy_lookup(x=0, y=0, z=0, w=roll))
            kin_dict["Roll CPy Migration Cached"]["RL"].append(self.RL_cpy_lookup(x=0, y=0, z=0, w=roll))
            kin_dict["Roll CPy Migration Cached"]["RR"].append(self.RR_cpy_lookup(x=0, y=0, z=0, w=roll))

            kin_dict["Roll Kw Cached"]["FL"].append(self.FL_Kw_lookup(x=0, y=0, z=0, w=roll))
            kin_dict["Roll Kw Cached"]["FR"].append(self.FR_Kw_lookup(x=0, y=0, z=0, w=roll))
            kin_dict["Roll Kw Cached"]["RL"].append(self.RL_Kw_lookup(x=0, y=0, z=0, w=roll))
            kin_dict["Roll Kw Cached"]["RR"].append(self.RR_Kw_lookup(x=0, y=0, z=0, w=roll))

            kin_dict["Roll Kr Cached"]["FL"].append(self.Fr_Kr_lookup(x=0, y=0, z=0, w=roll))
            kin_dict["Roll Kr Cached"]["FR"].append(self.Fr_Kr_lookup(x=0, y=0, z=0, w=roll))
            kin_dict["Roll Kr Cached"]["RL"].append(self.Rr_Kr_lookup(x=0, y=0, z=0, w=roll))
            kin_dict["Roll Kr Cached"]["RR"].append(self.Rr_Kr_lookup(x=0, y=0, z=0, w=roll))

            kin_dict["Roll Kp Cached"]["FL"].append(self.Kp_lookup(x=0, y=0, z=0, w=roll))
            kin_dict["Roll Kp Cached"]["FR"].append(self.Kp_lookup(x=0, y=0, z=0, w=roll))
            kin_dict["Roll Kp Cached"]["RL"].append(self.Kp_lookup(x=0, y=0, z=0, w=roll))
            kin_dict["Roll Kp Cached"]["RR"].append(self.Kp_lookup(x=0, y=0, z=0, w=roll))
        
        fig_1, ax_1 = plt.subplots(nrows=2, ncols=2)
        fig_1.set_size_inches(w=11, h=8.5)
        fig_1.suptitle("Bump Camber")
        fig_1.subplots_adjust(wspace=0.2, hspace=0.4)
        
        ax_1[0, 0].set_xlabel("Jounce (mm)")
        ax_1[0, 0].set_ylabel("Camber (deg)")
        ax_1[0, 0].set_title("FL Bump Camber")
        # ax_1[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Camber Lite"]["FL"])
        ax_1[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Camber Full"]["FL"])
        # ax_1[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Camber Cached"]["FL"])
        ax_1[0, 0].grid()

        ax_1[0, 1].set_xlabel("Jounce (mm)")
        ax_1[0, 1].set_ylabel("Camber (deg)")
        ax_1[0, 1].set_title("FR Bump Camber")
        # ax_1[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Camber Lite"]["FR"])
        ax_1[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Camber Full"]["FR"])
        # ax_1[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Camber Cached"]["FR"])
        ax_1[0, 1].grid()

        ax_1[1, 0].set_xlabel("Jounce (mm)")
        ax_1[1, 0].set_ylabel("Camber (deg)")
        ax_1[1, 0].set_title("RL Bump Camber")
        # ax_1[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Camber Lite"]["RL"])
        ax_1[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Camber Full"]["RL"])
        # ax_1[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Camber Cached"]["RL"])
        ax_1[1, 0].grid()

        ax_1[1, 1].set_xlabel("Jounce (mm)")
        ax_1[1, 1].set_ylabel("Camber (deg)")
        ax_1[1, 1].set_title("RR Bump Camber")
        # ax_1[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Camber Lite"]["RR"])
        ax_1[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Camber Full"]["RR"])
        # ax_1[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Camber Cached"]["RR"])
        # fig_1.legend(["Lite Model", "Full Model", "Cached Model"])
        ax_1[1, 1].grid()

        fig_2, ax_2 = plt.subplots(nrows=2, ncols=2)
        fig_2.set_size_inches(w=11, h=8.5)
        fig_2.suptitle("Bump Toe")
        fig_2.subplots_adjust(wspace=0.2, hspace=0.4)

        ax_2[0, 0].set_xlabel("Jounce (mm)")
        ax_2[0, 0].set_ylabel("Toe (deg)")
        ax_2[0, 0].set_title("FL Bump Toe")
        # ax_2[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Toe Lite"]["FL"])
        ax_2[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Toe Full"]["FL"])
        # ax_2[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Toe Cached"]["FL"])
        ax_2[0, 0].grid()

        ax_2[0, 1].set_xlabel("Jounce (mm)")
        ax_2[0, 1].set_ylabel("Toe (deg)")
        ax_2[0, 1].set_title("FR Bump Toe")
        # ax_2[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Toe Lite"]["FR"])
        ax_2[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Toe Full"]["FR"])
        # ax_2[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Toe Cached"]["FR"])
        ax_2[0, 1].grid()

        ax_2[1, 0].set_xlabel("Jounce (mm)")
        ax_2[1, 0].set_ylabel("Toe (deg)")
        ax_2[1, 0].set_title("RL Bump Toe")
        # ax_2[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Toe Lite"]["RL"])
        ax_2[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Toe Full"]["RL"])
        # ax_2[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Toe Cached"]["RL"])
        ax_2[1, 0].grid()

        ax_2[1, 1].set_xlabel("Jounce (mm)")
        ax_2[1, 1].set_ylabel("Toe (deg)")
        ax_2[1, 1].set_title("RR Bump Toe")
        # ax_2[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Toe Lite"]["RR"])
        ax_2[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Toe Full"]["RR"])
        # ax_2[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Toe Cached"]["RR"])
        # fig_2.legend(["Lite Model", "Full Model", "Cached Model"])
        ax_2[1, 1].grid()

        fig_3, ax_3 = plt.subplots(nrows=2, ncols=2)
        fig_3.set_size_inches(w=11, h=8.5)
        fig_3.suptitle("Bump Caster")
        fig_3.subplots_adjust(wspace=0.2, hspace=0.4)

        ax_3[0, 0].set_xlabel("Jounce (mm)")
        ax_3[0, 0].set_ylabel("Caster (deg)")
        ax_3[0, 0].set_title("FL Bump Caster")
        # ax_3[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Caster Lite"]["FL"])
        ax_3[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Caster Full"]["FL"])
        # ax_3[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Caster Cached"]["FL"])
        ax_3[0, 0].grid()

        ax_3[0, 1].set_xlabel("Jounce (mm)")
        ax_3[0, 1].set_ylabel("Caster (deg)")
        ax_3[0, 1].set_title("FR Bump Caster")
        # ax_3[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Caster Lite"]["FR"])
        ax_3[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Caster Full"]["FR"])
        # ax_3[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Caster Cached"]["FR"])
        ax_3[0, 1].grid()

        ax_3[1, 0].set_xlabel("Jounce (mm)")
        ax_3[1, 0].set_ylabel("Caster (deg)")
        ax_3[1, 0].set_title("RL Bump Caster")
        # ax_3[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Caster Lite"]["RL"])
        ax_3[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Caster Full"]["RL"])
        # ax_3[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Caster Cached"]["RL"])
        ax_3[1, 0].grid()

        ax_3[1, 1].set_xlabel("Jounce (mm)")
        ax_3[1, 1].set_ylabel("Caster (deg)")
        ax_3[1, 1].set_title("RR Bump Caster")
        # ax_3[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Caster Lite"]["RR"])
        ax_3[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Caster Full"]["RR"])
        # ax_3[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Caster Cached"]["RR"])
        # fig_3.legend(["Lite Model", "Full Model", "Cached Model"])
        ax_3[1, 1].grid()

        fig_4, ax_4 = plt.subplots(nrows=2, ncols=2)
        fig_4.set_size_inches(w=11, h=8.5)
        fig_4.suptitle("FVIC Bump Migration")
        fig_4.subplots_adjust(wspace=0.2, hspace=0.4)
        
        ax_4[0, 0].set_xlabel("Jounce (mm)")
        ax_4[0, 0].set_ylabel("FVIC Height (m)")
        ax_4[0, 0].set_title("FL FVIC Migration")
        # ax_4[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump FVIC Migration Lite"]["FL"])
        ax_4[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump FVIC Migration Full"]["FL"])
        # ax_4[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump FVIC Migration Cached"]["FL"])
        ax_4[0, 0].grid()

        ax_4[0, 1].set_xlabel("Jounce (mm)")
        ax_4[0, 1].set_ylabel("Caster (deg)")
        ax_4[0, 1].set_title("FR FVIC Migration")
        # ax_4[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump FVIC Migration Lite"]["FR"])
        ax_4[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump FVIC Migration Full"]["FR"])
        # ax_4[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump FVIC Migration Cached"]["FR"])
        ax_4[0, 1].grid()

        ax_4[1, 0].set_xlabel("Jounce (mm)")
        ax_4[1, 0].set_ylabel("FVIC Height (m)")
        ax_4[1, 0].set_title("RL FVIC Migration")
        # ax_4[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump FVIC Migration Lite"]["RL"])
        ax_4[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump FVIC Migration Full"]["RL"])
        # ax_4[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump FVIC Migration Cached"]["RL"])
        ax_4[1, 0].grid()

        ax_4[1, 1].set_xlabel("Jounce (mm)")
        ax_4[1, 1].set_ylabel("FVIC Height (m)")
        ax_4[1, 1].set_title("RR FVIC Migration")
        # ax_4[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump FVIC Migration Lite"]["RR"])
        ax_4[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump FVIC Migration Full"]["RR"])
        # ax_4[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump FVIC Migration Cached"]["RR"])
        # fig_4.legend(["Lite Model", "Full Model", "Cached Model"])
        ax_4[1, 1].grid()

        fig_5, ax_5 = plt.subplots(nrows=2, ncols=2)
        fig_5.set_size_inches(w=11, h=8.5)
        fig_5.suptitle("SVIC Bump Migration")
        fig_5.subplots_adjust(wspace=0.2, hspace=0.4)
        
        ax_5[0, 0].set_xlabel("Jounce (mm)")
        ax_5[0, 0].set_ylabel("SVIC Height (m)")
        ax_5[0, 0].set_title("FL SVIC Migration")
        # ax_5[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump SVIC Migration Lite"]["FL"])
        ax_5[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump SVIC Migration Full"]["FL"])
        # ax_5[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump SVIC Migration Cached"]["FL"])
        ax_5[0, 0].grid()

        ax_5[0, 1].set_xlabel("Jounce (mm)")
        ax_5[0, 1].set_ylabel("SVIC Height (m)")
        ax_5[0, 1].set_title("FL SVIC Migration")
        # ax_5[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump SVIC Migration Lite"]["FR"])
        ax_5[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump SVIC Migration Full"]["FR"])
        # ax_5[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump SVIC Migration Cached"]["FR"])
        ax_5[0, 1].grid()

        ax_5[1, 0].set_xlabel("Jounce (mm)")
        ax_5[1, 0].set_ylabel("SVIC Height (m)")
        ax_5[1, 0].set_title("RL SVIC Migration")
        # ax_5[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump SVIC Migration Lite"]["RL"])
        ax_5[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump SVIC Migration Full"]["RL"])
        # ax_5[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump SVIC Migration Cached"]["RL"])
        ax_5[1, 0].grid()

        ax_5[1, 1].set_xlabel("Jounce (mm)")
        ax_5[1, 1].set_ylabel("SVIC Height (m)")
        ax_5[1, 1].set_title("RR SVIC Migration")
        # ax_5[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump SVIC Migration Lite"]["RR"])
        ax_5[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump SVIC Migration Full"]["RR"])
        # ax_5[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump SVIC Migration Cached"]["RR"])
        # fig_5.legend(["Lite Model", "Full Model", "Cached Model"])
        ax_5[1, 1].grid()

        fig_6, ax_6 = plt.subplots(nrows=2, ncols=2)
        fig_6.set_size_inches(w=11, h=8.5)
        fig_6.suptitle("Bump FAPz Migration")
        fig_6.subplots_adjust(wspace=0.2, hspace=0.4)
        
        ax_6[0, 0].set_xlabel("Jounce (mm)")
        ax_6[0, 0].set_ylabel("FAPz (m)")
        ax_6[0, 0].set_title("FL FAPz Migration")
        # ax_6[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump FAPz Migration Lite"]["FL"])
        ax_6[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump FAPz Migration Full"]["FL"])
        # ax_6[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump FAPz Migration Cached"]["FL"])
        ax_6[0, 0].grid()

        ax_6[0, 1].set_xlabel("Jounce (mm)")
        ax_6[0, 1].set_ylabel("FAPz Height (m)")
        ax_6[0, 1].set_title("FR FAPz Migration")
        # ax_6[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump FAPz Migration Lite"]["FR"])
        ax_6[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump FAPz Migration Full"]["FR"])
        # ax_6[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump FAPz Migration Cached"]["FR"])
        ax_6[0, 1].grid()

        ax_6[1, 0].set_xlabel("Jounce (mm)")
        ax_6[1, 0].set_ylabel("FAPz Height (m)")
        ax_6[1, 0].set_title("RL FAPz Migration")
        # ax_6[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump FAPz Migration Lite"]["RL"])
        ax_6[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump FAPz Migration Full"]["RL"])
        # ax_6[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump FAPz Migration Cached"]["RL"])
        ax_6[1, 0].grid()

        ax_6[1, 1].set_xlabel("Jounce (mm)")
        ax_6[1, 1].set_ylabel("FAPz Height (m)")
        ax_6[1, 1].set_title("RR FAPz Migration")
        # ax_6[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump FAPz Migration Lite"]["RR"])
        ax_6[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump FAPz Migration Full"]["RR"])
        # ax_6[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump FAPz Migration Cached"]["RR"])
        # fig_6.legend(["Lite Model", "Full Model", "Cached Model"])
        ax_6[1, 1].grid()

        fig_7, ax_7 = plt.subplots(nrows=2, ncols=2)
        fig_7.set_size_inches(w=11, h=8.5)
        fig_7.suptitle("Bump CPx Migration")
        fig_7.subplots_adjust(wspace=0.2, hspace=0.4)
        
        ax_7[0, 0].set_xlabel("Jounce (mm)")
        ax_7[0, 0].set_ylabel("CPx (m)")
        ax_7[0, 0].set_title("FL CPx Migration")
        # ax_7[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump CPx Migration Lite"]["FL"])
        ax_7[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump CPx Migration Full"]["FL"])
        # ax_7[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump CPx Migration Cached"]["FL"])
        ax_7[0, 0].grid()

        ax_7[0, 1].set_xlabel("Jounce (mm)")
        ax_7[0, 1].set_ylabel("CPx (m)")
        ax_7[0, 1].set_title("FR CPx Migration")
        # ax_7[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump CPx Migration Lite"]["FR"])
        ax_7[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump CPx Migration Full"]["FR"])
        # ax_7[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump CPx Migration Cached"]["FR"])
        ax_7[0, 1].grid()

        ax_7[1, 0].set_xlabel("Jounce (mm)")
        ax_7[1, 0].set_ylabel("CPx (m)")
        ax_7[1, 0].set_title("RL CPx Migration")
        # ax_7[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump CPx Migration Lite"]["RL"])
        ax_7[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump CPx Migration Full"]["RL"])
        # ax_7[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump CPx Migration Cached"]["RL"])
        ax_7[1, 0].grid()

        ax_7[1, 1].set_xlabel("Jounce (mm)")
        ax_7[1, 1].set_ylabel("CPx (m)")
        ax_7[1, 1].set_title("RR CPx Migration")
        # ax_7[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump CPx Migration Lite"]["RR"])
        ax_7[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump CPx Migration Full"]["RR"])
        # ax_7[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump CPx Migration Cached"]["RR"])
        # fig_7.legend(["Lite Model", "Full Model", "Cached Model"])
        ax_7[1, 1].grid()

        fig_8, ax_8 = plt.subplots(nrows=2, ncols=2)
        fig_8.set_size_inches(w=11, h=8.5)
        fig_8.suptitle("Bump CPx Migration")
        fig_8.subplots_adjust(wspace=0.2, hspace=0.4)
        
        ax_8[0, 0].set_xlabel("Jounce (mm)")
        ax_8[0, 0].set_ylabel("CPy (m)")
        ax_8[0, 0].set_title("FL CPy Migration")
        # ax_8[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump CPy Migration Lite"]["FL"])
        ax_8[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump CPy Migration Full"]["FL"])
        # ax_8[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump CPy Migration Cached"]["FL"])
        ax_8[0, 0].grid()

        ax_8[0, 1].set_xlabel("Jounce (mm)")
        ax_8[0, 1].set_ylabel("CPy (m)")
        ax_8[0, 1].set_title("FR CPy Migration")
        # ax_8[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump CPy Migration Lite"]["FR"])
        ax_8[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump CPy Migration Full"]["FR"])
        # ax_8[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump CPy Migration Cached"]["FR"])
        ax_8[0, 1].grid()

        ax_8[1, 0].set_xlabel("Jounce (mm)")
        ax_8[1, 0].set_ylabel("CPy (m)")
        ax_8[1, 0].set_title("RL CPy Migration")
        # ax_8[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump CPy Migration Lite"]["RL"])
        ax_8[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump CPy Migration Full"]["RL"])
        # ax_8[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump CPy Migration Cached"]["RL"])
        ax_8[1, 0].grid()

        ax_8[1, 1].set_xlabel("Jounce (mm)")
        ax_8[1, 1].set_ylabel("CPy (m)")
        ax_8[1, 1].set_title("RR CPy Migration")
        # ax_8[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump CPy Migration Lite"]["RR"])
        ax_8[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump CPy Migration Full"]["RR"])
        # ax_8[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump CPy Migration Cached"]["RR"])
        fig_8.legend(["Lite Model", "Full Model", "Cached Model"])
        ax_8[1, 1].grid()

        fig_9, ax_9 = plt.subplots(nrows=2, ncols=2)
        fig_9.set_size_inches(w=11, h=8.5)
        fig_9.suptitle("Bump Kw")
        fig_9.subplots_adjust(wspace=0.2, hspace=0.4)
        
        ax_9[0, 0].set_xlabel("Jounce (mm)")
        ax_9[0, 0].set_ylabel("Kw (N/m)")
        ax_9[0, 0].set_title("FL Kw")
        # ax_9[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Kw Lite"]["FL"])
        ax_9[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Kw Full"]["FL"])
        # ax_9[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Kw Cached"]["FL"])

        ax_9[0, 1].set_xlabel("Jounce (mm)")
        ax_9[0, 1].set_ylabel("Kw (N/m)")
        ax_9[0, 1].set_title("FR Kw")
        # ax_9[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Kw Lite"]["FR"])
        ax_9[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Kw Full"]["FR"])
        # ax_9[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Kw Cached"]["FR"])

        ax_9[1, 0].set_xlabel("Jounce (mm)")
        ax_9[1, 0].set_ylabel("Kw (N/m)")
        ax_9[1, 0].set_title("RL Kw")
        # ax_9[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Kw Lite"]["RL"])
        ax_9[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Kw Full"]["RL"])
        # ax_9[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Kw Cached"]["RL"])

        ax_9[1, 1].set_xlabel("Jounce (mm)")
        ax_9[1, 1].set_ylabel("Kw (N/m)")
        ax_9[1, 1].set_title("RR Kw")
        # ax_9[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Kw Lite"]["RR"])
        ax_9[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Kw Full"]["RR"])
        # ax_9[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Kw Cached"]["RR"])
        fig_9.legend(["Lite Model", "Full Model", "Cached Model"])

        fig_10, ax_10 = plt.subplots(nrows=2, ncols=2)
        fig_10.set_size_inches(w=11, h=8.5)
        fig_10.suptitle("Bump Kr")
        fig_10.subplots_adjust(wspace=0.2, hspace=0.4)
        
        ax_10[0, 0].set_xlabel("Jounce (mm)")
        ax_10[0, 0].set_ylabel("Kr (Nm/rad)")
        ax_10[0, 0].set_title("FL Kr")
        # ax_10[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Kr Lite"]["FL"])
        ax_10[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Kr Full"]["FL"])
        # ax_10[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Kr Cached"]["FL"])

        ax_10[0, 1].set_xlabel("Jounce (mm)")
        ax_10[0, 1].set_ylabel("Kr (Nm/rad)")
        ax_10[0, 1].set_title("FR Kr")
        # ax_10[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Kr Lite"]["FR"])
        ax_10[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Kr Full"]["FR"])
        # ax_10[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Kr Cached"]["FR"])

        ax_10[1, 0].set_xlabel("Jounce (mm)")
        ax_10[1, 0].set_ylabel("Kr (Nm/rad)")
        ax_10[1, 0].set_title("RL Kr")
        # ax_10[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Kr Lite"]["RL"])
        ax_10[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Kr Full"]["RL"])
        # ax_10[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Kr Cached"]["RL"])

        ax_10[1, 1].set_xlabel("Jounce (mm)")
        ax_10[1, 1].set_ylabel("Kr (Nm/rad)")
        ax_10[1, 1].set_title("RR Kr")
        # ax_10[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Kr Lite"]["RR"])
        ax_10[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Kr Full"]["RR"])
        # ax_10[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Kr Cached"]["RR"])
        fig_10.legend(["Lite Model", "Full Model", "Cached Model"])

        fig_11, ax_11 = plt.subplots(nrows=2, ncols=2)
        fig_11.set_size_inches(w=11, h=8.5)
        fig_11.suptitle("Bump Kp")
        fig_11.subplots_adjust(wspace=0.2, hspace=0.4)
        
        ax_11[0, 0].set_xlabel("Jounce (mm)")
        ax_11[0, 0].set_ylabel("Kp (Nm/rad)")
        ax_11[0, 0].set_title("FL Kp")
        # ax_11[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Kp Lite"]["FL"])
        ax_11[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Kp Full"]["FL"])
        # ax_11[0, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Kp Cached"]["FL"])

        ax_11[0, 1].set_xlabel("Jounce (mm)")
        ax_11[0, 1].set_ylabel("Kp (Nm/rad)")
        ax_11[0, 1].set_title("FR Kp")
        # ax_11[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Kp Lite"]["FR"])
        ax_11[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Kp Full"]["FR"])
        # ax_11[0, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Kp Cached"]["FR"])

        ax_11[1, 0].set_xlabel("Jounce (mm)")
        ax_11[1, 0].set_ylabel("Kp (Nm/rad)")
        ax_11[1, 0].set_title("RL Kp")
        # ax_11[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Kp Lite"]["RL"])
        ax_11[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Kp Full"]["RL"])
        # ax_11[1, 0].plot(self.heave_sweep * 1000, kin_dict["Bump Kp Cached"]["RL"])

        ax_11[1, 1].set_xlabel("Jounce (mm)")
        ax_11[1, 1].set_ylabel("Kp (Nm/rad)")
        ax_11[1, 1].set_title("RR Kp")
        # ax_11[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Kp Lite"]["RR"])
        ax_11[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Kp Full"]["RR"])
        # ax_11[1, 1].plot(self.heave_sweep * 1000, kin_dict["Bump Kp Cached"]["RR"])
        fig_11.legend(["Lite Model", "Full Model", "Cached Model"])

        fig_12, ax_12 = plt.subplots(nrows=2, ncols=2)
        fig_12.set_size_inches(w=11, h=8.5)
        fig_12.suptitle("Roll Camber")
        fig_12.subplots_adjust(wspace=0.2, hspace=0.4)
        
        ax_12[0, 0].set_xlabel("Roll (deg)")
        ax_12[0, 0].set_ylabel("Camber (deg)")
        ax_12[0, 0].set_title("FL Roll Camber")
        # ax_12[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Camber Lite"]["FL"])
        ax_12[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Camber Full"]["FL"])
        # ax_12[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Camber Cached"]["FL"])

        ax_12[0, 1].set_xlabel("Roll (deg)")
        ax_12[0, 1].set_ylabel("Camber (deg)")
        ax_12[0, 1].set_title("FR Roll Camber")
        # ax_12[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Camber Lite"]["FR"])
        ax_12[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Camber Full"]["FR"])
        # ax_12[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Camber Cached"]["FR"])

        ax_12[1, 0].set_xlabel("Roll (deg)")
        ax_12[1, 0].set_ylabel("Camber (deg)")
        ax_12[1, 0].set_title("RL Roll Camber")
        # ax_12[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Camber Lite"]["RL"])
        ax_12[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Camber Full"]["RL"])
        # ax_12[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Camber Cached"]["RL"])

        ax_12[1, 1].set_xlabel("Roll (deg)")
        ax_12[1, 1].set_ylabel("Camber (deg)")
        ax_12[1, 1].set_title("RR Roll Camber")
        # ax_12[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Camber Lite"]["RR"])
        ax_12[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Camber Full"]["RR"])
        # ax_12[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Camber Cached"]["RR"])
        fig_12.legend(["Lite Model", "Full Model", "Cached Model"])

        fig_13, ax_13 = plt.subplots(nrows=2, ncols=2)
        fig_13.set_size_inches(w=11, h=8.5)
        fig_13.suptitle("Roll Toe")
        fig_13.subplots_adjust(wspace=0.2, hspace=0.4)
        
        ax_13[0, 0].set_xlabel("Roll (deg)")
        ax_13[0, 0].set_ylabel("Toe (deg)")
        ax_13[0, 0].set_title("FL Roll Toe")
        # ax_13[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Toe Lite"]["FL"])
        ax_13[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Toe Full"]["FL"])
        # ax_13[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Toe Cached"]["FL"])

        ax_13[0, 1].set_xlabel("Roll (deg)")
        ax_13[0, 1].set_ylabel("Toe (deg)")
        ax_13[0, 1].set_title("FR Roll Toe")
        # ax_13[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Toe Lite"]["FR"])
        ax_13[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Toe Full"]["FR"])
        # ax_13[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Toe Cached"]["FR"])

        ax_13[1, 0].set_xlabel("Roll (deg)")
        ax_13[1, 0].set_ylabel("Toe (deg)")
        ax_13[1, 0].set_title("RL Roll Toe")
        # ax_13[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Toe Lite"]["RL"])
        ax_13[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Toe Full"]["RL"])
        # ax_13[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Toe Cached"]["RL"])

        ax_13[1, 1].set_xlabel("Roll (deg)")
        ax_13[1, 1].set_ylabel("Toe (deg)")
        ax_13[1, 1].set_title("RR Roll Toe")
        # ax_13[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Toe Lite"]["RR"])
        ax_13[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Toe Full"]["RR"])
        # ax_13[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Toe Cached"]["RR"])
        fig_13.legend(["Lite Model", "Full Model", "Cached Model"])

        fig_14, ax_14 = plt.subplots(nrows=2, ncols=2)
        fig_14.set_size_inches(w=11, h=8.5)
        fig_14.suptitle("Roll Caster")
        fig_14.subplots_adjust(wspace=0.2, hspace=0.4)
        
        ax_14[0, 0].set_xlabel("Roll (deg)")
        ax_14[0, 0].set_ylabel("Caster (deg)")
        ax_14[0, 0].set_title("FL Roll Caster")
        # ax_14[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Caster Lite"]["FL"])
        ax_14[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Caster Full"]["FL"])
        # ax_14[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Caster Cached"]["FL"])

        ax_14[0, 1].set_xlabel("Roll (deg)")
        ax_14[0, 1].set_ylabel("Caster (deg)")
        ax_14[0, 1].set_title("FR Roll Caster")
        # ax_14[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Caster Lite"]["FR"])
        ax_14[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Caster Full"]["FR"])
        # ax_14[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Caster Cached"]["FR"])

        ax_14[1, 0].set_xlabel("Roll (deg)")
        ax_14[1, 0].set_ylabel("Caster (deg)")
        ax_14[1, 0].set_title("RL Roll Caster")
        # ax_14[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Caster Lite"]["RL"])
        ax_14[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Caster Full"]["RL"])
        # ax_14[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Caster Cached"]["RL"])

        ax_14[1, 1].set_xlabel("Roll (deg)")
        ax_14[1, 1].set_ylabel("Caster (deg)")
        ax_14[1, 1].set_title("RR Roll Caster")
        # ax_14[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Caster Lite"]["RR"])
        ax_14[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Caster Full"]["RR"])
        # ax_14[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Caster Cached"]["RR"])
        fig_14.legend(["Lite Model", "Full Model", "Cached Model"])

        fig_15, ax_15 = plt.subplots(nrows=2, ncols=2)
        fig_15.set_size_inches(w=11, h=8.5)
        fig_15.suptitle("FVIC Roll Migration")
        fig_15.subplots_adjust(wspace=0.2, hspace=0.4)
        
        ax_15[0, 0].set_xlabel("Roll (deg)")
        ax_15[0, 0].set_ylabel("FVIC Height (m)")
        ax_15[0, 0].set_title("FL FVIC Migration")
        # ax_15[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll FVIC Migration Lite"]["FL"])
        ax_15[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll FVIC Migration Full"]["FL"])
        # ax_15[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll FVIC Migration Cached"]["FL"])

        ax_15[0, 1].set_xlabel("Roll (deg)")
        ax_15[0, 1].set_ylabel("FVIC Height (m)")
        ax_15[0, 1].set_title("FR FVIC Migration")
        # ax_15[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll FVIC Migration Lite"]["FR"])
        ax_15[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll FVIC Migration Full"]["FR"])
        # ax_15[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll FVIC Migration Cached"]["FR"])

        ax_15[1, 0].set_xlabel("Roll (deg)")
        ax_15[1, 0].set_ylabel("FVIC Height (m)")
        ax_15[1, 0].set_title("RL FVIC Migration")
        # ax_15[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll FVIC Migration Lite"]["RL"])
        ax_15[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll FVIC Migration Full"]["RL"])
        # ax_15[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll FVIC Migration Cached"]["RL"])

        ax_15[1, 1].set_xlabel("Roll (deg)")
        ax_15[1, 1].set_ylabel("FVIC Height (m)")
        ax_15[1, 1].set_title("RR FVIC Migration")
        # ax_15[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll FVIC Migration Lite"]["RR"])
        ax_15[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll FVIC Migration Full"]["RR"])
        # ax_15[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll FVIC Migration Cached"]["RR"])
        fig_15.legend(["Lite Model", "Full Model", "Cached Model"])

        fig_16, ax_16 = plt.subplots(nrows=2, ncols=2)
        fig_16.set_size_inches(w=11, h=8.5)
        fig_16.suptitle("SVIC Roll Migration")
        fig_16.subplots_adjust(wspace=0.2, hspace=0.4)
        
        ax_16[0, 0].set_xlabel("Roll (deg)")
        ax_16[0, 0].set_ylabel("SVIC Height (m)")
        ax_16[0, 0].set_title("FL SVIC Migration")
        # ax_16[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll SVIC Migration Lite"]["FL"])
        ax_16[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll SVIC Migration Full"]["FL"])
        # ax_16[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll SVIC Migration Cached"]["FL"])

        ax_16[0, 1].set_xlabel("Roll (deg)")
        ax_16[0, 1].set_ylabel("FVIC Height (m)")
        ax_16[0, 1].set_title("FR SVIC Migration")
        # ax_16[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll SVIC Migration Lite"]["FR"])
        ax_16[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll SVIC Migration Full"]["FR"])
        # ax_16[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll SVIC Migration Cached"]["FR"])

        ax_16[1, 0].set_xlabel("Roll (deg)")
        ax_16[1, 0].set_ylabel("SVIC Height (m)")
        ax_16[1, 0].set_title("RL SVIC Migration")
        # ax_16[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll SVIC Migration Lite"]["RL"])
        ax_16[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll SVIC Migration Full"]["RL"])
        # ax_16[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll SVIC Migration Cached"]["RL"])

        ax_16[1, 1].set_xlabel("Roll (deg)")
        ax_16[1, 1].set_ylabel("SVIC Height (m)")
        ax_16[1, 1].set_title("RR SVIC Migration")
        # ax_16[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll SVIC Migration Lite"]["RR"])
        ax_16[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll SVIC Migration Full"]["RR"])
        # ax_16[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll SVIC Migration Cached"]["RR"])
        fig_16.legend(["Lite Model", "Full Model", "Cached Model"])

        fig_17, ax_17 = plt.subplots(nrows=2, ncols=2)
        fig_17.set_size_inches(w=11, h=8.5)
        fig_17.suptitle("Roll FAPz Migration")
        fig_17.subplots_adjust(wspace=0.2, hspace=0.4)
        
        ax_17[0, 0].set_xlabel("Roll (deg)")
        ax_17[0, 0].set_ylabel("FAPz (m)")
        ax_17[0, 0].set_title("FL FAPz Migration")
        # ax_17[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll FAPz Migration Lite"]["FL"])
        ax_17[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll FAPz Migration Full"]["FL"])
        # ax_17[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll FAPz Migration Cached"]["FL"])

        ax_17[0, 1].set_xlabel("Roll (deg)")
        ax_17[0, 1].set_ylabel("FAPz Height (m)")
        ax_17[0, 1].set_title("FR FAPz Migration")
        # ax_17[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll FAPz Migration Lite"]["FR"])
        ax_17[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll FAPz Migration Full"]["FR"])
        # ax_17[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll FAPz Migration Cached"]["FR"])

        ax_17[1, 0].set_xlabel("Roll (deg)")
        ax_17[1, 0].set_ylabel("FAPz Height (m)")
        ax_17[1, 0].set_title("RL FAPz Migration")
        # ax_17[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll FAPz Migration Lite"]["RL"])
        ax_17[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll FAPz Migration Full"]["RL"])
        # ax_17[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll FAPz Migration Cached"]["RL"])

        ax_17[1, 1].set_xlabel("Roll (deg)")
        ax_17[1, 1].set_ylabel("FAPz Height (m)")
        ax_17[1, 1].set_title("RR FAPz Migration")
        # ax_17[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll FAPz Migration Lite"]["RR"])
        ax_17[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll FAPz Migration Full"]["RR"])
        # ax_17[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll FAPz Migration Cached"]["RR"])
        fig_17.legend(["Lite Model", "Full Model", "Cached Model"])

        fig_18, ax_18 = plt.subplots(nrows=2, ncols=2)
        fig_18.set_size_inches(w=11, h=8.5)
        fig_18.suptitle("Roll CPx Migration")
        fig_18.subplots_adjust(wspace=0.2, hspace=0.4)
        
        ax_18[0, 0].set_xlabel("Roll (deg)")
        ax_18[0, 0].set_ylabel("CPx (m)")
        ax_18[0, 0].set_title("FL CPx Migration")
        # ax_18[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll CPx Migration Lite"]["FL"])
        ax_18[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll CPx Migration Full"]["FL"])
        # ax_18[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll CPx Migration Cached"]["FL"])

        ax_18[0, 1].set_xlabel("Roll (deg)")
        ax_18[0, 1].set_ylabel("CPx (m)")
        ax_18[0, 1].set_title("FR CPx Migration")
        # ax_18[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll CPx Migration Lite"]["FR"])
        ax_18[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll CPx Migration Full"]["FR"])
        # ax_18[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll CPx Migration Cached"]["FR"])

        ax_18[1, 0].set_xlabel("Roll (deg)")
        ax_18[1, 0].set_ylabel("CPx (m)")
        ax_18[1, 0].set_title("RL CPx Migration")
        # ax_18[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll CPx Migration Lite"]["RL"])
        ax_18[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll CPx Migration Full"]["RL"])
        # ax_18[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll CPx Migration Cached"]["RL"])

        ax_18[1, 1].set_xlabel("Roll (deg)")
        ax_18[1, 1].set_ylabel("CPx (m)")
        ax_18[1, 1].set_title("RR CPx Migration")
        # ax_18[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll CPx Migration Lite"]["RR"])
        ax_18[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll CPx Migration Full"]["RR"])
        # ax_18[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll CPx Migration Cached"]["RR"])
        fig_18.legend(["Lite Model", "Full Model", "Cached Model"])

        fig_19, ax_19 = plt.subplots(nrows=2, ncols=2)
        fig_19.set_size_inches(w=11, h=8.5)
        fig_19.suptitle("Roll CPx Migration")
        fig_19.subplots_adjust(wspace=0.2, hspace=0.4)
        
        ax_19[0, 0].set_xlabel("Roll (deg)")
        ax_19[0, 0].set_ylabel("CPy (m)")
        ax_19[0, 0].set_title("FL CPy Migration")
        # ax_19[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll CPy Migration Lite"]["FL"])
        ax_19[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll CPy Migration Full"]["FL"])
        # ax_19[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll CPy Migration Cached"]["FL"])

        ax_19[0, 1].set_xlabel("Roll (deg)")
        ax_19[0, 1].set_ylabel("CPy (m)")
        ax_19[0, 1].set_title("FR CPy Migration")
        # ax_19[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll CPy Migration Lite"]["FR"])
        ax_19[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll CPy Migration Full"]["FR"])
        # ax_19[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll CPy Migration Cached"]["FR"])

        ax_19[1, 0].set_xlabel("Roll (deg)")
        ax_19[1, 0].set_ylabel("CPy (m)")
        ax_19[1, 0].set_title("RL CPy Migration")
        # ax_19[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll CPy Migration Lite"]["RL"])
        ax_19[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll CPy Migration Full"]["RL"])
        # ax_19[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll CPy Migration Cached"]["RL"])

        ax_19[1, 1].set_xlabel("Roll (deg)")
        ax_19[1, 1].set_ylabel("CPy (m)")
        ax_19[1, 1].set_title("RR CPy Migration")
        # ax_19[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll CPy Migration Lite"]["RR"])
        ax_19[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll CPy Migration Full"]["RR"])
        # ax_19[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll CPy Migration Cached"]["RR"])
        fig_19.legend(["Lite Model", "Full Model", "Cached Model"])

        fig_20, ax_20 = plt.subplots(nrows=2, ncols=2)
        fig_20.set_size_inches(w=11, h=8.5)
        fig_20.suptitle("Roll Kw")
        fig_20.subplots_adjust(wspace=0.2, hspace=0.4)
        
        ax_20[0, 0].set_xlabel("Roll (deg)")
        ax_20[0, 0].set_ylabel("Kw (N/m)")
        ax_20[0, 0].set_title("FL Kw")
        # ax_20[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kw Lite"]["FL"])
        ax_20[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kw Full"]["FL"])
        # ax_20[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kw Cached"]["FL"])

        ax_20[0, 1].set_xlabel("Roll (deg)")
        ax_20[0, 1].set_ylabel("Kw (N/m)")
        ax_20[0, 1].set_title("FR Kw")
        # ax_20[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kw Lite"]["FR"])
        ax_20[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kw Full"]["FR"])
        # ax_20[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kw Cached"]["FR"])

        ax_20[1, 0].set_xlabel("Roll (deg)")
        ax_20[1, 0].set_ylabel("Kw (N/m)")
        ax_20[1, 0].set_title("RL Kw")
        # ax_20[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kw Lite"]["RL"])
        ax_20[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kw Full"]["RL"])
        # ax_20[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kw Cached"]["RL"])

        ax_20[1, 1].set_xlabel("Roll (deg)")
        ax_20[1, 1].set_ylabel("Kw (N/m)")
        ax_20[1, 1].set_title("RR Kw")
        # ax_20[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kw Lite"]["RR"])
        ax_20[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kw Full"]["RR"])
        # ax_20[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kw Cached"]["RR"])
        fig_20.legend(["Lite Model", "Full Model", "Cached Model"])

        fig_21, ax_21 = plt.subplots(nrows=2, ncols=2)
        fig_21.set_size_inches(w=11, h=8.5)
        fig_21.suptitle("Roll Kr")
        fig_21.subplots_adjust(wspace=0.2, hspace=0.4)
        
        ax_21[0, 0].set_xlabel("Roll (deg)")
        ax_21[0, 0].set_ylabel("Kr (Nm/rad)")
        ax_21[0, 0].set_title("FL Kr")
        # ax_21[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kr Lite"]["FL"])
        ax_21[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kr Full"]["FL"])
        # ax_21[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kr Cached"]["FL"])

        ax_21[0, 1].set_xlabel("Roll (deg)")
        ax_21[0, 1].set_ylabel("Kr (Nm/rad)")
        ax_21[0, 1].set_title("FR Kr")
        # ax_21[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kr Lite"]["FR"])
        ax_21[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kr Full"]["FR"])
        # ax_21[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kr Cached"]["FR"])

        ax_21[1, 0].set_xlabel("Roll (deg)")
        ax_21[1, 0].set_ylabel("Kr (Nm/rad)")
        ax_21[1, 0].set_title("RL Kr")
        # ax_21[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kr Lite"]["RL"])
        ax_21[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kr Full"]["RL"])
        # ax_21[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kr Cached"]["RL"])

        ax_21[1, 1].set_xlabel("Roll (deg)")
        ax_21[1, 1].set_ylabel("Kr (Nm/rad)")
        ax_21[1, 1].set_title("RR Kr")
        # ax_21[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kr Lite"]["RR"])
        ax_21[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kr Full"]["RR"])
        # ax_21[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kr Cached"]["RR"])
        fig_21.legend(["Lite Model", "Full Model", "Cached Model"])

        fig_22, ax_22 = plt.subplots(nrows=2, ncols=2)
        fig_22.set_size_inches(w=11, h=8.5)
        fig_22.suptitle("Roll Kp")
        fig_22.subplots_adjust(wspace=0.2, hspace=0.4)
        
        ax_22[0, 0].set_xlabel("Roll (deg)")
        ax_22[0, 0].set_ylabel("Kp (Nm/rad)")
        ax_22[0, 0].set_title("FL Kp")
        # ax_22[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kp Lite"]["FL"])
        ax_22[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kp Full"]["FL"])
        # ax_22[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kp Cached"]["FL"])

        ax_22[0, 1].set_xlabel("Roll (deg)")
        ax_22[0, 1].set_ylabel("Kp (Nm/rad)")
        ax_22[0, 1].set_title("FR Kp")
        # ax_22[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kp Lite"]["FR"])
        ax_22[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kp Full"]["FR"])
        # ax_22[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kp Cached"]["FR"])

        ax_22[1, 0].set_xlabel("Roll (deg)")
        ax_22[1, 0].set_ylabel("Kp (Nm/rad)")
        ax_22[1, 0].set_title("RL Kp")
        # ax_22[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kp Lite"]["RL"])
        ax_22[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kp Full"]["RL"])
        # ax_22[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kp Cached"]["RL"])

        ax_22[1, 1].set_xlabel("Roll (deg)")
        ax_22[1, 1].set_ylabel("Kp (Nm/rad)")
        ax_22[1, 1].set_title("RR Kp")
        # ax_22[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kp Lite"]["RR"])
        ax_22[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kp Full"]["RR"])
        # ax_22[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Kp Cached"]["RR"])
        fig_22.legend(["Lite Model", "Full Model", "Cached Model"])

        fig_23, ax_23 = plt.subplots(nrows=2, ncols=2)
        fig_23.set_size_inches(w=11, h=8.5)
        fig_23.suptitle("Steer Camber")
        fig_23.subplots_adjust(wspace=0.2, hspace=0.4)
        
        ax_23[0, 0].set_xlabel("Steer (deg)")
        ax_23[0, 0].set_ylabel("Camber (deg)")
        ax_23[0, 0].set_title("FL Steer Camber")
        ax_23[0, 0].plot(self.steer_sweep / (3.50 / 360 * 0.0254), kin_dict["Steer Camber Full_-2"]["FL"])
        ax_23[0, 0].plot(self.steer_sweep / (3.50 / 360 * 0.0254), kin_dict["Steer Camber Full_-1"]["FL"])
        ax_23[0, 0].plot(self.steer_sweep / (3.50 / 360 * 0.0254), kin_dict["Steer Camber Full"]["FL"])
        ax_23[0, 0].plot(self.steer_sweep / (3.50 / 360 * 0.0254), kin_dict["Steer Camber Full_1"]["FL"])
        ax_23[0, 0].plot(self.steer_sweep / (3.50 / 360 * 0.0254), kin_dict["Steer Camber Full_2"]["FL"])
        
        ax_23[0, 1].set_xlabel("Steer (deg)")
        ax_23[0, 1].set_ylabel("Camber (deg)")
        ax_23[0, 1].set_title("FR Steer Camber")
        ax_23[0, 1].plot(self.steer_sweep / (3.50 / 360 * 0.0254), kin_dict["Steer Camber Full_-2"]["FR"])
        ax_23[0, 1].plot(self.steer_sweep / (3.50 / 360 * 0.0254), kin_dict["Steer Camber Full_-1"]["FR"])
        ax_23[0, 1].plot(self.steer_sweep / (3.50 / 360 * 0.0254), kin_dict["Steer Camber Full"]["FR"])
        ax_23[0, 1].plot(self.steer_sweep / (3.50 / 360 * 0.0254), kin_dict["Steer Camber Full_1"]["FR"])
        ax_23[0, 1].plot(self.steer_sweep / (3.50 / 360 * 0.0254), kin_dict["Steer Camber Full_2"]["FR"])

        ax_23[1, 0].set_xlabel("Steer (deg)")
        ax_23[1, 0].set_ylabel("Camber (deg)")
        ax_23[1, 0].set_title("RL Steer Camber")
        ax_23[1, 0].plot(self.steer_sweep / (3.50 / 360 * 0.0254), kin_dict["Steer Camber Full_-2"]["RL"])
        ax_23[1, 0].plot(self.steer_sweep / (3.50 / 360 * 0.0254), kin_dict["Steer Camber Full_-1"]["RL"])
        ax_23[1, 0].plot(self.steer_sweep / (3.50 / 360 * 0.0254), kin_dict["Steer Camber Full"]["RL"])
        ax_23[1, 0].plot(self.steer_sweep / (3.50 / 360 * 0.0254), kin_dict["Steer Camber Full_1"]["RL"])
        ax_23[1, 0].plot(self.steer_sweep / (3.50 / 360 * 0.0254), kin_dict["Steer Camber Full_2"]["RL"])

        ax_23[1, 1].set_xlabel("Steer (deg)")
        ax_23[1, 1].set_ylabel("Camber (deg)")
        ax_23[1, 1].set_title("RR Steer Camber")
        ax_23[1, 1].plot(self.steer_sweep / (3.50 / 360 * 0.0254), kin_dict["Steer Camber Full_-2"]["RR"])
        ax_23[1, 1].plot(self.steer_sweep / (3.50 / 360 * 0.0254), kin_dict["Steer Camber Full_-1"]["RR"])
        ax_23[1, 1].plot(self.steer_sweep / (3.50 / 360 * 0.0254), kin_dict["Steer Camber Full"]["RR"])
        ax_23[1, 1].plot(self.steer_sweep / (3.50 / 360 * 0.0254), kin_dict["Steer Camber Full_1"]["RR"])
        ax_23[1, 1].plot(self.steer_sweep / (3.50 / 360 * 0.0254), kin_dict["Steer Camber Full_2"]["RR"])
        
        ax_23[0, 0].legend(["-2 deg", "-1 deg", "0 deg", "1 deg", "2 deg"])
        ax_23[0, 1].legend(["-2 deg", "-1 deg", "0 deg", "1 deg", "2 deg"])
        ax_23[1, 0].legend(["-2 deg", "-1 deg", "0 deg", "1 deg", "2 deg"])
        ax_23[1, 1].legend(["-2 deg", "-1 deg", "0 deg", "1 deg", "2 deg"])
        
        fig_23.legend(["Full Model"])

        fig_24, ax_24 = plt.subplots(nrows=2, ncols=2)
        fig_24.set_size_inches(w=11, h=8.5)
        fig_24.suptitle("Steer Toe")
        fig_24.subplots_adjust(wspace=0.2, hspace=0.4)
        
        ax_24[0, 0].set_xlabel("Steer (deg)")
        ax_24[0, 0].set_ylabel("Toe (deg)")
        ax_24[0, 0].set_title("FL Steer Toe")
        ax_24[0, 0].plot(self.steer_sweep / (3.50 / 360 * 0.0254), kin_dict["Steer Toe Full"]["FL"])

        ax_24[0, 1].set_xlabel("Steer (deg)")
        ax_24[0, 1].set_ylabel("Toe (deg)")
        ax_24[0, 1].set_title("FR Steer Toe")
        ax_24[0, 1].plot(self.steer_sweep / (3.50 / 360 * 0.0254), kin_dict["Steer Toe Full"]["FR"])

        ax_24[1, 0].set_xlabel("Steer (deg)")
        ax_24[1, 0].set_ylabel("Toe (deg)")
        ax_24[1, 0].set_title("RL Steer Toe")
        ax_24[1, 0].plot(self.steer_sweep / (3.50 / 360 * 0.0254), kin_dict["Steer Toe Full"]["RL"])

        ax_24[1, 1].set_xlabel("Steer (deg)")
        ax_24[1, 1].set_ylabel("Toe (deg)")
        ax_24[1, 1].set_title("RR Steer Toe")
        ax_24[1, 1].plot(self.steer_sweep / (3.50 / 360 * 0.0254), kin_dict["Steer Toe Full"]["RR"])
        fig_24.legend(["Full Model"])

        fig_25, ax_25 = plt.subplots(nrows=2, ncols=2)
        fig_25.set_size_inches(w=11, h=8.5)
        fig_25.suptitle("Steer Caster")
        fig_25.subplots_adjust(wspace=0.2, hspace=0.4)
        
        ax_25[0, 0].set_xlabel("Steer (deg)")
        ax_25[0, 0].set_ylabel("Caster (deg)")
        ax_25[0, 0].set_title("FL Steer Caster")
        ax_25[0, 0].plot(self.steer_sweep / (3.50 / 360 * 0.0254), kin_dict["Steer Caster Full"]["FL"])

        ax_25[0, 1].set_xlabel("Steer (deg)")
        ax_25[0, 1].set_ylabel("Caster (deg)")
        ax_25[0, 1].set_title("FR Steer Caster")
        ax_25[0, 1].plot(self.steer_sweep / (3.50 / 360 * 0.0254), kin_dict["Steer Caster Full"]["FR"])

        ax_25[1, 0].set_xlabel("Steer (deg)")
        ax_25[1, 0].set_ylabel("Caster (deg)")
        ax_25[1, 0].set_title("RL Steer Caster")
        ax_25[1, 0].plot(self.steer_sweep / (3.50 / 360 * 0.0254), kin_dict["Steer Caster Full"]["RL"])

        ax_25[1, 1].set_xlabel("Steer (deg)")
        ax_25[1, 1].set_ylabel("Caster (deg)")
        ax_25[1, 1].set_title("RR Steer Caster")
        ax_25[1, 1].plot(self.steer_sweep / (3.50 / 360 * 0.0254), kin_dict["Steer Caster Full"]["RR"])
        fig_25.legend(["Full Model"])

        fig_26, ax_26 = plt.subplots(nrows=2, ncols=2)
        fig_26.set_size_inches(w=11, h=8.5)
        fig_26.suptitle("Roll KinRC z Migration")
        fig_26.subplots_adjust(wspace=0.2, hspace=0.4)
        
        ax_26[0, 0].set_xlabel("Roll (deg)")
        ax_26[0, 0].set_ylabel("KinRC Height(m)")
        ax_26[0, 0].set_title("FL KinRC z Migration")
        # ax_26[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll KinRC z Migration Lite"]["FL"])
        ax_26[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll KinRC z Migration Full"]["FL"])
        # ax_26[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll KinRC z Migration Cached"]["FL"])
        ax_26[0, 0].grid()

        ax_26[0, 1].set_xlabel("Roll (deg)")
        ax_26[0, 1].set_ylabel("KinRC Height (m)")
        ax_26[0, 1].set_title("FR KinRC z Migration")
        # ax_26[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll KinRC z Migration Lite"]["FR"])
        ax_26[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll KinRC z Migration Full"]["FR"])
        # ax_26[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll KinRC z Migration Cached"]["FR"])
        ax_26[0, 1].grid()

        ax_26[1, 0].set_xlabel("Roll (deg)")
        ax_26[1, 0].set_ylabel("KinRC Height (m)")
        ax_26[1, 0].set_title("RL KinRC z Migration")
        # ax_26[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll KinRC z Migration Lite"]["RL"])
        ax_26[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll KinRC z Migration Full"]["RL"])
        # ax_26[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll KinRC z Migration Cached"]["RL"])
        ax_26[1, 0].grid()

        ax_26[1, 1].set_xlabel("Roll (deg)")
        ax_26[1, 1].set_ylabel("KinRC Height (m)")
        ax_26[1, 1].set_title("RR KinRC z Migration")
        # ax_26[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll KinRC z Migration Lite"]["RR"])
        ax_26[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll KinRC z Migration Full"]["RR"])
        # ax_26[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll KinRC z Migration Cached"]["RR"])
        ax_26[1, 1].grid()
        # fig_26.legend(["Lite Model", "Full Model", "Cached Model"])

        fig_27, ax_27 = plt.subplots(nrows=2, ncols=2)
        fig_27.set_size_inches(w=11, h=8.5)
        fig_27.suptitle("Roll KinRC y Migration")
        fig_27.subplots_adjust(wspace=0.2, hspace=0.4)
        
        ax_27[0, 0].set_xlabel("Roll (deg)")
        ax_27[0, 0].set_ylabel("KinRC Height(m)")
        ax_27[0, 0].set_title("FL KinRC y Migration")
        # ax_27[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll KinRC y Migration Lite"]["FL"])
        ax_27[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll KinRC y Migration Full"]["FL"])
        # ax_27[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll KinRC y Migration Cached"]["FL"])
        ax_27[0, 0].grid()

        ax_27[0, 1].set_xlabel("Roll (deg)")
        ax_27[0, 1].set_ylabel("KinRC Height (m)")
        ax_27[0, 1].set_title("FR KinRC y Migration")
        # ax_27[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll KinRC y Migration Lite"]["FR"])
        ax_27[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll KinRC y Migration Full"]["FR"])
        # ax_27[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll KinRC y Migration Cached"]["FR"])
        ax_27[0, 1].grid()

        ax_27[1, 0].set_xlabel("Roll (deg)")
        ax_27[1, 0].set_ylabel("KinRC Height (m)")
        ax_27[1, 0].set_title("RL KinRC y Migration")
        # ax_27[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll KinRC y Migration Lite"]["RL"])
        ax_27[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll KinRC y Migration Full"]["RL"])
        # ax_27[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll KinRC y Migration Cached"]["RL"])
        ax_27[1, 0].grid()

        ax_27[1, 1].set_xlabel("Roll (deg)")
        ax_27[1, 1].set_ylabel("KinRC Height (m)")
        ax_27[1, 1].set_title("RR KinRC y Migration")
        # ax_27[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll KinRC y Migration Lite"]["RR"])
        ax_27[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll KinRC y Migration Full"]["RR"])
        # ax_26[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll KinRC y Migration Cached"]["RR"])
        ax_27[1, 1].grid()
        # fig_26.legend(["Lite Model", "Full Model", "Cached Model"])

        fig_28, ax_28 = plt.subplots(nrows=2, ncols=2)
        fig_28.set_size_inches(w=11, h=8.5)
        fig_28.suptitle("Roll Scrub")
        fig_28.subplots_adjust(wspace=0.2, hspace=0.4)
        
        ax_28[0, 0].set_xlabel("Roll (deg)")
        ax_28[0, 0].set_ylabel("Scrub (mm)")
        ax_28[0, 0].set_title("FL Scrub")
        # ax_28[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Scrub Lite"]["FL"])
        ax_28[0, 0].plot(self.roll_sweep * 180 / np.pi, np.array(kin_dict["Roll Scrub Full"]["FL"]) * 1000)
        # ax_28[0, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Scrub Cached"]["FL"])
        ax_28[0, 0].grid()

        ax_28[0, 1].set_xlabel("Roll (deg)")
        ax_28[0, 1].set_ylabel("Scrub (mm)")
        ax_28[0, 1].set_title("FR Scrub")
        # ax_28[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Scrub Lite"]["FR"])
        ax_28[0, 1].plot(self.roll_sweep * 180 / np.pi, np.array(kin_dict["Roll Scrub Full"]["FR"]) * 1000)
        # ax_28[0, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Scrub Cached"]["FR"])
        ax_28[0, 1].grid()

        ax_28[1, 0].set_xlabel("Roll (deg)")
        ax_28[1, 0].set_ylabel("Scrub (mm)")
        ax_28[1, 0].set_title("RL Scrub")
        # ax_28[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Scrub Lite"]["RL"])
        ax_28[1, 0].plot(self.roll_sweep * 180 / np.pi, np.array(kin_dict["Roll Scrub Full"]["RL"]) * 1000)
        # ax_28[1, 0].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Scrub Cached"]["RL"])
        ax_28[1, 0].grid()

        ax_28[1, 1].set_xlabel("Roll (deg)")
        ax_28[1, 1].set_ylabel("Scrub (mm)")
        ax_28[1, 1].set_title("RR Scrub")
        # ax_28[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Scrub Lite"]["RR"])
        ax_28[1, 1].plot(self.roll_sweep * 180 / np.pi, np.array(kin_dict["Roll Scrub Full"]["RR"]) * 1000)
        # ax_26[1, 1].plot(self.roll_sweep * 180 / np.pi, kin_dict["Roll Scrub Cached"]["RR"])
        ax_28[1, 1].grid()
        # fig_26.legend(["Lite Model", "Full Model", "Cached Model"])

        self.plot(figs=[fig_1, fig_2, fig_3, fig_4, fig_5, fig_6, fig_7, fig_8, fig_9, fig_10, fig_11,
                        fig_12, fig_13, fig_14, fig_15, fig_16, fig_17, fig_18, fig_19, fig_20, fig_21, 
                        fig_22, fig_23, fig_24, fig_25, fig_26, fig_27, fig_28])
    
    def generate_kin_helper(self, steer: float, heave: float, pitch: float, roll: float, reset: bool = False):
        dist_vecs = {"FL": {"CGx": None, "CGy": None},
                     "FR": {"CGx": None, "CGy": None},
                     "RL": {"CGx": None, "CGy": None},
                     "RR": {"CGx": None, "CGy": None}}
        sus_corners = [self.FL_double_wishbone, self.FR_double_wishbone, self.RL_double_wishbone, self.RR_double_wishbone]

        for i, key in enumerate(dist_vecs.keys()):
            x_dist = abs(sus_corners[i].contact_patch.position[0] - self.cg.position[0])
            y_dist = abs(sus_corners[i].contact_patch.position[1] - self.cg.position[1])
            dist_vecs[key]["CGx"] = x_dist
            dist_vecs[key]["CGy"] = y_dist

        FL_jounce = heave + dist_vecs["FL"]["CGx"] * np.tan(pitch) - dist_vecs["FL"]["CGy"] * np.tan(roll)
        FR_jounce = heave + dist_vecs["FR"]["CGx"] * np.tan(pitch) + dist_vecs["FR"]["CGy"] * np.tan(roll)
        RL_jounce = heave - dist_vecs["RL"]["CGx"] * np.tan(pitch) - dist_vecs["RL"]["CGy"] * np.tan(roll)
        RR_jounce = heave - dist_vecs["RR"]["CGx"] * np.tan(pitch) + dist_vecs["RR"]["CGy"] * np.tan(roll)

        self.FL_dw.steer(steer=steer)
        self.FR_dw.steer(steer=steer)

        self.FL_dw.jounce(jounce=FL_jounce)
        self.FR_dw.jounce(jounce=FR_jounce)
        self.RL_dw.jounce(jounce=RL_jounce)
        self.RR_dw.jounce(jounce=RR_jounce)

    def plot(self, figs: Sequence[Figure]) -> Figure:
        p = PdfPages("./outputs/kin_plots.pdf")

        for page in figs:
            page.savefig(p, format="pdf")

        p.close()