# rocket_model.py
from rocketpy import Environment, SolidMotor, Rocket, Flight
import os

DATA_DIR = "./data"

MOTOR_FILE = os.path.join(
    DATA_DIR, "motors", "cesaroni", "Cesaroni_M1670.eng"
)
POWER_OFF_DRAG_FILE = os.path.join(
    DATA_DIR, "rockets", "calisto", "powerOffDragCurve.csv"
)
POWER_ON_DRAG_FILE = os.path.join(
    DATA_DIR, "rockets", "calisto", "powerOnDragCurve.csv"
)

def build_environment():
    """
    Crea un ambiente RocketPy per il sito EuroC.
    """
    env = Environment(
        latitude=39.3897,
        longitude=-8.28897,
        elevation=160,  # m ASL
    )
    env.set_date((2025, 10, 15, 12))
    env.set_atmospheric_model(type="standard_atmosphere")
    return env

def build_rocket(env, mass_scale=1.0):
    """
    Crea razzo tipo Calisto. mass_scale ti permette di
    introdurre piccole variazioni di massa tra i voli.
    """
    motor = SolidMotor(
        thrust_source=MOTOR_FILE,
        dry_mass=1.815,
        dry_inertia=(0.125, 0.125, 0.002),
        nozzle_radius=33 / 1000,
        grain_number=5,
        grain_density=1815,
        grain_outer_radius=33 / 1000,
        grain_initial_inner_radius=15 / 1000,
        grain_initial_height=120 / 1000,
        grain_separation=5 / 1000,
        grains_center_of_mass_position=0.397,
        center_of_dry_mass_position=0.317,
        nozzle_position=0,
        burn_time=3.9,
        throat_radius=11 / 1000,
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )

    rocket = Rocket(
        radius=127 / 2000,
        mass=14.426 * mass_scale,  # massa senza motore scalata
        inertia=(6.321, 6.321, 0.034),
        power_off_drag=POWER_OFF_DRAG_FILE,
        power_on_drag=POWER_ON_DRAG_FILE,
        center_of_mass_without_motor=0,
        coordinate_system_orientation="tail_to_nose",
    )

    rocket.set_rail_buttons(
        upper_button_position=0.0818,
        lower_button_position=-0.618,
        angular_position=45,
    )

    rocket.add_motor(motor, position=-1.255)

    rocket.add_nose(
        length=0.55829,
        kind="vonKarman",
        position=1.278,
    )

    rocket.add_trapezoidal_fins(
        n=4,
        root_chord=0.120,
        tip_chord=0.060,
        span=0.110,
        position=-1.04956,
        cant_angle=0.5,
        airfoil=(os.path.join(DATA_DIR, "airfoils", "NACA0012-radians.txt"), "radians"),
    )

    rocket.add_tail(
        top_radius=0.0635,
        bottom_radius=0.0435,
        length=0.060,
        position=-1.194656,
    )

    return rocket, motor

def simulate_flight(mass_scale=1.0, rail_inclination_deg=84, wind_u=0.0, wind_v=0.0):
    """
    Esegue una simulazione RocketPy e ritorna (env, rocket, motor, flight).
    Puoi randomizzare mass_scale, inclinazione, vento per generare dataset vari.
    """
    env = build_environment()

    # modello di vento semplice personalizzato
    if wind_u != 0.0 or wind_v != 0.0:
        env.set_atmospheric_model(
            type="custom_atmosphere",
            wind_u=lambda h: wind_u,
            wind_v=lambda h: wind_v,
        )

    rocket, motor = build_rocket(env, mass_scale=mass_scale)

    flight = Flight(
        rocket=rocket,
        environment=env,
        rail_length=5.2,
        inclination=rail_inclination_deg,
        heading=133,
        time_overshoot=True,
        terminate_on_apogee=False,  # vogliamo tutta la traiettoria
    )

    return env, rocket, motor, flight
