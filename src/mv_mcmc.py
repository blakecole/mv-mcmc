# ********************************************************** #
#    NAME: Blake Cole                                        #
#    ORGN: (self)                                            #
#    FILE: mv_mcmc.py                                        #
#    DATE: 28 FEB 2025                                       #
# ********************************************************** #

from scipy.io import loadmat
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geotools as gt


# =============================================================================
# Swim Simulation
# =============================================================================
def blind_swim(t_start, dt, lon_start, lat_start, speed=1.0, max_t=3.0):
    """
    Simulate a trajectory through a vector field for a constant heading.
    It tests headings from 90° to 275° (step 5°) and selects the one that reaches
    the shoreline in the minimum elapsed time.

    Parameters:
        t_start (float): Initial time (in hours).
        dt (float): Time step (in hours).
        lon_start (float): Starting longitude.
        lat_start (float): Starting latitude.
        speed (float): Swimming speed.
        max_t (float): Maximum allowed simulation time (in hours).

    Returns:
        numpy.ndarray: An array filled with the best constant heading repeated for each step,
                       or None if no heading reaches shore.
    """
    dt_seconds = dt * 3600  # precompute time step in seconds
    best_heading = None
    best_time = float("inf")
    best_steps = None
    viable_trajectories = []
    viable_times = []

    # Sweep over constant southbound headings
    for heading in range(90, 280, 5):
        current_loc = (lon_start, lat_start)
        trajectory = [current_loc]
        current_time = t_start
        steps = 0
        shore_hits = 0
        lost_at_sea = False

        # Precompute the constant swim velocity for this heading.
        heading_rad = np.radians(heading)
        swim_velocity = speed * np.array([np.sin(heading_rad), np.cos(heading_rad)])
        
        # Continue simulation until we have at least one shoreline hits.
        while shore_hits < 2:
            steps += 1
            
            # Compute flow field and update velocity over ground.
            flow_velocity = flowfield(current_loc[0], current_loc[1], current_time)
            velocity = flow_velocity + swim_velocity

            # Calculate course over ground (COG) and speed over ground (SOG)
            cog = np.degrees(np.mod(np.pi / 2 - np.arctan2(np.radians(velocity[1]),
                                                            np.radians(velocity[0])), 2*np.pi))
            sog = np.linalg.norm(velocity)
            distance = sog * dt_seconds

            # Update position and time.
            current_loc = gt.reckon(current_loc, cog, distance)
            current_time += dt
            trajectory.append(current_loc)
            
            # Increment shore hit count if the current point is in the region.
            if gt.is_point_in_region(current_loc, mv_shoreline):
                shore_hits += 1

            elapsed_time_sec = (current_time - t_start) * 3600
            if elapsed_time_sec > max_t * 3600:
                lost_at_sea = True
                break
    
        if lost_at_sea:
            continue
        
        # If trajectory hits land, save trajectory and time.
        viable_trajectories.append(trajectory)
        viable_times.append(elapsed_time_sec/60)
        
        if elapsed_time_sec < best_time:
            print(f"Land ho! Constant heading {heading} reached land in {int(elapsed_time_sec)} seconds.")
            best_heading = heading
            best_time = elapsed_time_sec
            best_steps = steps

    if best_heading is None:
        print("No valid heading found!")
        return None
    
    # Plot candidate 'constant heading' trajectories
    num_lines = len(viable_trajectories)
    norm = mcolors.PowerNorm(gamma=0.7, vmin=np.min(viable_times), vmax=np.max(viable_times))
    for i in range(num_lines):
        traj = np.asarray(viable_trajectories[i])
        time = np.asarray(viable_times[i])
        c = plt.cm.bone_r(norm(time))
        ax.plot(traj[:,0], traj[:,1], color=c, linewidth=0.5)
    
    # Create a ScalarMappable with the same colormap and normalization for the colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.bone_r, norm=norm)
    sm.set_array([])  # Dummy array needed for ScalarMappable
    
    # Add the colorbar to the plot
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Swim Time [min]')
    headings = np.full(best_steps, float(best_heading))
    print(f"Best Time: {best_time:.0f} seconds")
    print(f"Best Heading: {best_heading:.0f} degrees")
    print()
    return headings


def simulate_swim(t_start, dt, lon_start, lat_start, headings, speed=1.0):
    # Compute the trajectory of a swim defined by:
    #   - a start time, specified in hours from the beginning of the dataset
    #   - a starting point (loc_start), defined as a (lon, lat) tuple
    #   - a list of headings (trajectory)
    #   - a predetermined time interval (dt) between heading changes.
    h = np.radians(headings)
    loc = np.array([lon_start, lat_start])
    t = t_start
    
    # for storage
    n_headings = np.size(headings)
    time = np.zeros((n_headings+1, 1))
    traj = np.zeros((n_headings+1, 2))
    
    # Store variables
    time[0] = t
    traj[0,:] = loc

    
    for i in range(n_headings):
        # Compute velocity over ground at present location
        flow = flowfield(loc[0], loc[1], t)
        swim = speed * np.array([np.sin(h[i]), np.cos(h[i])])
        vog = flow + swim
        
        # Compute the distance over ground covered in one time step
        cog = np.degrees(np.mod(np.pi/2 - np.atan2(np.radians(vog[1]), np.radians(vog[0])), 2*np.pi))
        sog = np.sqrt(np.dot(vog, vog))
        dist = sog*dt*3600
        
        # Compute new position after timestep, increment time
        loc = gt.reckon(loc, cog, dist)
        t += dt
        
        # Store variables
        time[i+1] = t
        traj[i+1,:] = loc

    total_time = (t-t_start)*3600
    
    return total_time, traj


# =============================================================================
# Cost Function
# =============================================================================
def cost_fn(headings):
    # Wrapper for swim time; helpful when defining a more complex cost function.
    total_time, traj = simulate_swim(t_start, dt, lon_start, lat_start, headings, swim_speed_ms)
    
    ## Option: penalize jagged trajectories
    # dlon = np.diff(traj[:,0], n=2, axis=0)
    # dlat = np.diff(traj[:,1], n=2, axis=0)
    # smoothness = np.sum(dlon**2 + dlat**2)
    
    cost = total_time
    return cost
        
        
# =============================================================================
# RJMCMC Proposal: Birth, Death, and Modify Moves
# =============================================================================
def generate_bump(headings, bump_size=1.0):
    # Random center in [-0.2, 1.2]
    center = np.random.uniform(-0.2, 1.2)
    
    # Random width: generate log-width uniformly between log(0.05) and log(1), then exponentiate
    log_width = np.random.uniform(np.log(0.05), np.log(1.0))
    width = np.exp(log_width)
    
    # Random amplitude in [-bump_size, bump_size]
    amplitude = np.random.uniform(-bump_size, bump_size)
    
    # Create sample points: the number of points equals the length of headings
    n_points = len(headings)
    x_vals = np.linspace(0, 1, n_points)
    
    # Evaluate the Gaussian bump at each sample point
    bump = amplitude * np.exp(-((x_vals - center) ** 2) / (width ** 2))
    
    return bump


def propose_rjmcmc(headings, bump_size=1.0):
    """
    Proposes a new candidate for the heading sequence h using RJMCMC moves.
    Moves include:
      - 'modify': Adjust an existing heading (dimension unchanged)
      - 'birth': Add a new heading (increases dimension)
      - 'death': Remove a heading (decreases dimension)
    """
    birth_bearing = float(initial_h[0])
    new_h = headings.copy()
    pb = 0.33
    pd = 0.33
    pm = 0.34
    move_type = np.random.choice(['birth', 'death', 'modify'], p=[pb, pd, pm])
    
    # For each move, record any needed information for the reverse move.
    if move_type == 'modify' and len(new_h) > 0:
        perturbation = generate_bump(new_h, bump_size)
        new_h += perturbation
        move_info = {'type': 'modify'}
    
    elif move_type == 'birth':
        new_heading = np.random.normal(birth_bearing, 20.0)
        insert_idx = len(new_h)
        new_h = np.insert(new_h, insert_idx, new_heading)
        move_info = {'type': 'birth', 'insert_idx': insert_idx, 'new_heading': new_heading}
    
    elif move_type == 'death' and len(new_h) > 1:
        #idx = np.random.randint(len(new_h))
        idx = len(new_h)-1
        removed_heading = new_h[idx]
        new_h = np.delete(new_h, idx)
        move_info = {'type': 'death', 'remove_idx': idx, 'removed_heading': removed_heading}
    
    else:
        # Fallback: if a death move isn't possible (e.g., h is too short), perform a modify.
        if len(new_h) > 0:
            perturbation = generate_bump(new_h, bump_size)
            new_h += perturbation
            move_info = {'type': 'modify'}
        else:
            # If empty, force a birth move.
            new_heading = np.random.uniform(0, 359.9)
            new_h.append(new_heading)
            move_info = {'type': 'birth', 'insert_idx': 0, 'new_heading': new_heading}
    
    return new_h, move_info


# =============================================================================
# RJMCMC Optimization Routine
# =============================================================================
def rjmcmc_optimize(initial_h, iterations=1000, temperature=1.0):
    """
    RJMCMC optimization that accounts for moves which change the dimensionality.
    For simplicity, we assume symmetric proposals (i.e. the probability of the reverse move is equal)
    and a Jacobian of 1 for the birth and death moves.
    In a rigorous application, you would calculate the proposal ratios explicitly.
    """
    
    # Ensure no overshoot in initial solution
    total_time, traj = simulate_swim(t_start, dt, lon_start, lat_start, initial_h, swim_speed_ms)
    chop_idx = np.where(gt.is_point_in_region(traj, mv_shoreline))[0]
    if chop_idx.size > 0:
        first_index = chop_idx[0]
        # Keep entries up to and including the condition-met element
        current_h = initial_h[:first_index+1]
        traj = traj[:first_index+1,:]
    else:
        current_h = initial_h
            
    current_cost = cost_fn(current_h)
    best_h = current_h
    best_cost = current_cost
    
    bump_size = 5.0
    
    init_temp = temperature
    final_temp = 1
    
    accepted_moves = 0
    total_moves = 0
    
    #to plot convergence lines
    #saved_traj = []
    #saved_time = []
    
    for i in range(iterations):
        # Cool bump size exponentially
        bump_size *= 0.999
        
        candidate_h, move_info = propose_rjmcmc(current_h, bump_size)
        total_time, traj = simulate_swim(t_start, dt, lon_start, lat_start, candidate_h, swim_speed_ms)
        candidate_cost = cost_fn(candidate_h)
        delta = candidate_cost - current_cost
        
        # For RJMCMC, compute the proposal ratio q(current -> candidate) / q(candidate -> current)
        # We assume symmetric proposals, so proposal_ratio = 1.
        proposal_ratio = 1.0
        
        # Compute acceptance probability w/ linear cooling
        temperature -= (init_temp - final_temp)/iterations
        acceptance_prob = min(1.0, np.exp(-delta / temperature) * proposal_ratio)
        
        if (delta < 0 or np.random.rand() < acceptance_prob) and gt.is_point_in_region(traj[-1,:], mv_shoreline):
            current_h = candidate_h
            current_cost = candidate_cost
            accepted_moves += 1
            
            if candidate_cost < best_cost:
                best_h = candidate_h
                best_cost = candidate_cost
        
        total_moves += 1
        
        if (i % 100 == 0):
            #saved_traj.append(traj)
            #saved_time.append(total_time)
            total_time, traj = simulate_swim(t_start, dt, lon_start, lat_start, best_h, swim_speed_ms)
            dist_to_shore, bearing_to_shore = gt.closest_appropach_to_region(traj[-1,:], mv_shoreline)
            print(f"Iteration {i}: Current Cost = {current_cost:.1f}, Best Cost = {best_cost:.1f}")
            print(f"Dist to shore = {dist_to_shore:.1f}")
            print(f"Total time (min) = {total_time/60:.2f}")
            print(f"Avg. Acceptance Probability (Temp) = {100*accepted_moves/total_moves:.2f}% ({temperature:.1f})")
            print(f"Bump Size = +/- {bump_size:.1f} degrees")
            accepted_moves = 0
            total_moves = 0
    
    # Plot candidate 'constant heading' trajectories
    #num_lines = len(saved_traj)
    #norm = mcolors.PowerNorm(gamma=0.7, vmin=np.min(saved_time), vmax=np.max(saved_time))
    #for i in range(num_lines):
    #    traj = np.asarray(saved_traj[i])
    #    time = np.asarray(saved_time[i])
    #    c = plt.cm.bone_r(norm(time))
    #    ax.plot(traj[:,0], traj[:,1], color=c, linewidth=0.5)
    
    # Create a ScalarMappable with the same colormap and normalization for the colorbar
    #sm = plt.cm.ScalarMappable(cmap=plt.cm.bone_r, norm=norm)
    #sm.set_array([])  # Dummy array needed for ScalarMappable
    
    # Add the colorbar to the plot
    #cbar = plt.colorbar(sm, ax=ax)
    #cbar.set_label('Swim Time [min]')
    return best_h, best_cost


# =============================================================================
# Data Setup
# =============================================================================
# Import land boundary files
print("Loading WH boundary data... ")
wh_shoreline = np.array(gt.read_kml_points('../data/wh_shoreline.kml'))
print("Success.")

print("Loading MV boundary data... ")
mv_shoreline = np.array(gt.read_kml_points('../data/mv_shoreline.kml'))
print("Success.")

# Import tidal current data (.mat -> numpy arrays)
print("Loading tidal current data... ")
data = loadmat('../data/tidal_data_2022_07_30.mat')
u = data['U_store']
v = data['V_store']
t_local_str = data['glocals']
t_local = [datetime.strptime(dt_str, "%Y-%b-%d %H:%M") for dt_str in t_local_str]
t = np.squeeze(data['thours'])
lat = data['lat']
lon = data['lon']
waypts = data['waypts']
print("Success.")
print()

# Create velocity field interpolation function
[u_interp_fn, v_interp_fn] = gt.make_flowfield_interp_fns(lon, lat, t, u, v, "nearest")
def flowfield(lon, lat, time):
    uq = u_interp_fn(lon, lat, time)
    vq = v_interp_fn(lon, lat, time)
    flow_vector = np.array([uq, vq])
    return flow_vector

# Configure Pyplot
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['CMU Serif']
plt.rcParams['figure.dpi'] = 600
plt.style.use('dark_background')

# Plot computational domain
fig, ax = plt.subplots()
ax.scatter(lon, lat, color='teal', s=0.1, label="Tidal Data")
ax.plot(wh_shoreline[:,0], wh_shoreline[:,1], color='red', linewidth=1.0, label="WH Shoreline")
ax.plot(mv_shoreline[:,0], mv_shoreline[:,1], color='green', linewidth=1.0, label="MV Shoreline")
ax.scatter(waypts[0,0], waypts[0,1], facecolors='none', edgecolors='red', s=20, label="Start")
ax.grid(alpha=0.5)
ax.axis('square')
ax.set_xlim(-70.70, -70.58)
ax.set_ylim(41.44, 41.55)
ax.set_xlabel("Longitude [deg]")
ax.set_ylabel("Latitude [deg]")


# =============================================================================
# MAIN EXECUTION: Run RJMCMC Optimization
# =============================================================================
if __name__ == '__main__':
    # Swim parameters
    swim_pace = 1.667                                           #[1:40 min/100yd]
    swim_speed_ms = 1/((swim_pace/100)*1760*(1/1609.34)*60)     #[m/s]
    lon_start = waypts[0,0]
    lat_start = waypts[0,1]
    dt = 0.005
    datetime_start = datetime.strptime("2022-Jul-30 06:30", "%Y-%b-%d %H:%M")
    
    # Optimization parameters
    iterations = 2000
    temperature = 15.0
    
    # Convert start time to 'datetime' object
    def hours_since_start(t_local_dt):
        return (t_local_dt - t_local[0]).total_seconds() / 3600
    x = np.array([hours_since_start(date_time) for date_time in t_local])
    target_x = hours_since_start(datetime_start)
    t_start =  np.round(np.interp(target_x, x, t), decimals=2)
    datetime_start_str = datetime_start.strftime("%Y-%b-%d %H:%M")

        
    # OPTIMIZE!
    # Search for a decent (constant-heading) first guess
    print("START: ", datetime_start, " ( t=", t_start, "hrs )")
    print("Searching for a reasonable first solution...")
    initial_h = blind_swim(t_start, dt, lon_start, lat_start, swim_speed_ms, max_t=3.0)
    #initial_h = np.full(int(3/dt), 180.0)
    
    # RJ-MCMC optimization routine
    optimized_h, optimized_cost = rjmcmc_optimize(initial_h, iterations, temperature)
    optimized_swim_time, optimized_swim_traj = simulate_swim(t_start, dt, lon_start, lat_start, optimized_h, swim_speed_ms)
    optimized_swim_time /= 60
    optimized_swim_dist = np.sum(gt.haversine_distance(optimized_swim_traj[0:-2,:], optimized_swim_traj[1:-1,:]))
    print("\nOptimized Headings:")
    print(np.round(optimized_h, decimals=2))
    print()
    print(f"\nOptimized Total Transit Time: {optimized_swim_time:.3f}")
    
    
    # PLOT!
    ax.set_title("Vineyard Sound Swim Optimization" + "\n" + datetime_start_str)
    ax.plot(optimized_swim_traj[:,0], optimized_swim_traj[:,1], color='gold', linewidth=1.0)
    
    # Print optimal swim stats in upper righthand corner.
    ax.text(
        0.76, 0.91,
        f'{optimized_swim_time:.1f} min\n{optimized_swim_dist:.0f} m',
        color='gold',        # Sets text color to gold
        fontsize=12,         # Optional: control text size
        ha='left',         # Horizontal alignment
        va='center',         # Vertical alignment
        transform=ax.transAxes,  # Positions text relative to Axes coordinates
        bbox=dict(
            facecolor='black',   # Background color of the box
            edgecolor='gold',    # Outline (edge) color of the box
            boxstyle='round,pad=0.2'  # Rounded box style, with some padding
        )
    )
    plt.tight_layout()
    plt.show()
        
    
    # SAVE!
    save_choice = input("Would you like to save the data? (y/n): ")
    if save_choice.lower().startswith('y'):
        fname_out = "mv_swim_heading_" + datetime_start.strftime("%H%M")+ ".csv"
        fpath_out = "../out/" + fname_out
        np.savetxt(fpath_out, optimized_h, delimiter=",", header="heading", fmt='%.1f')
        
        fname_out = "mv_swim_trajectory_" + datetime_start.strftime("%H%M")+ ".csv"
        fpath_out = "../out/" + fname_out
        np.savetxt(fpath_out, optimized_swim_traj, delimiter=",", header="longitude,latitude", fmt='%.6f')
        print("Data saved to: " + fpath_out)
    else:
        print("Data not saved.")
    
    # save figure
    save_choice = input("Would you like to save the figure? (y/n): ")
    if save_choice.lower().startswith('y'):
        fname_out = "mv_swim_" + datetime_start.strftime("%H%M")+ ".png"
        fpath_out = "../fig/" + fname_out
        fig.savefig(fpath_out, dpi=600, bbox_inches='tight')
        print("Figure saved to: " + fpath_out)
    else:
        print("Figure not saved.")