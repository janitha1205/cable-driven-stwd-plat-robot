import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def stewart_inverse_kinematics(pos, rpy, base_joints,base_jointst, platform_joints):

    # 1. Extract position coordinates
    x, y, z = pos
    T = np.array([x, y, z])
    p_base=np.zeros((3,3))
    # 2. Extract Euler angles (Roll, Pitch, Yaw)
    
    c=np.pi/180
    r, p, y = rpy
    # 3. Compute Rotation Matrix (R_z * R_y * R_x)
    R_x = np.array([
        [1, 0,           0          ],
        [0, np.cos(r),  -np.sin(r)  ],
        [0, np.sin(r),   np.cos(r)  ]
    ])
    
    R_y = np.array([
        [np.cos(p),  0,  np.sin(p)],
        [0,          1,  0        ],
        [-np.sin(p), 0,  np.cos(p)]
    ])
    
    R_z = np.array([
        [np.cos(y), -np.sin(y), 0],
        [np.sin(y),  np.cos(y), 0],
        [0,          0,         1]
    ])
    
    # Combined rotation matrix
    R = R_z @ R_y @ R_x
    
    # 4. Calculate required leg vectors and lengths
    leg_lengths = []
    
    for i in range(3):
        # Transform platform joint coordinate from local frame to base frame
        p_base[i] = T + (R @ platform_joints[i])
        
        # Vector from base joint to transformed platform joint
        leg_vector = p_base[i] - base_joints[i]
        leg_vector2= p_base[i] - base_jointst[i]
        
        # Leg length is the Euclidean norm of the vector
        leg_lengths.append( np.linalg.norm(leg_vector))
        leg_lengths.append( np.linalg.norm(leg_vector2))
        
    return leg_lengths,p_base


if __name__ == "__main__":
    # Define geometry: Radius of base and platform anchor circles
    r_base = 1.0      # 100 cm radius/1 m
    r_platform = 0.4    # 30 cm radiu/ 0.3 m
    
    # Semi-symmetrical angular positions for the 6 joints (in degrees)
    # Typically joints are arranged in 3 close pairs
    base_angles = np.degrees([0,180,300])
    platform_angles = np.degrees([0, 180, 300, 300])
    
    # Generate 3D coordinates for base and platform joints (Z is 0 in local frames)
    base_nodes = np.zeros((3, 3))
    base_nodest = np.zeros((3, 3))
    platform_nodes = np.zeros((6, 3))
    for i in range(3):
     base_nodes[i] = [r_base * np.cos(base_angles[i]), r_base * np.sin(base_angles[i]), 0.0]
     base_nodest[i] = [r_base * np.cos(base_angles[i]), r_base * np.sin(base_angles[i]), 1.0]
    for i in range(3):      
        platform_nodes[i] = [r_platform * np.cos(platform_angles[i]), r_platform * np.sin(platform_angles[i]), 0.0]
        
    # Define Target Pose
    target_position = [0.10, 0.20, 0.15]              # Center the platform at Z = 80 cm
    target_orientation = [0, np.pi/4, 0]   # Roll=10°, Pitch=5°, Yaw=0°
    
    # Compute inverse kinematics
    lengths,pb = stewart_inverse_kinematics(target_position, target_orientation, base_nodes,base_nodest, platform_nodes)
    print(pb)
    # Print out results
    print("--- Required Leg Lengths (Meters) ---")
    for idx, length in enumerate(lengths):
        print(f"Actuator Leg {idx + 1}: {length:.4f} m")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    base_plt = np.vstack([pb[0,:], pb[1,:], pb[2,:], pb[0,:]])
    base_leg1=[]
    base_leg2=[]
    for i in range(3):
        base_leg1=np.vstack([pb[i,:], base_nodes[i,:]])
        base_leg2=np.vstack([pb[i,:], base_nodest[i,:]])
        print(base_leg1)
        ax.plot(
            base_leg1[:,0],
            base_leg1[:,1],
            base_leg1[:,2],
            
            "k-",
            linewidth=1,
            label="Base",
        )    
        print(base_leg2)
        ax.plot(
            base_leg2[:,0],
            base_leg2[:,1],
            base_leg2[:,2],
            
            "k-",
            linewidth=1,
            label="Base",
        )  

    base_plot = np.vstack([base_nodes[0,:], base_nodes[1,:], base_nodes[2,:], base_nodes[0,:]])
    base_plot2 = np.vstack([base_nodest[0,:], base_nodest[1,:], base_nodest[2,:], base_nodest[0,:]])
    base_link1 = np.vstack([base_nodes[0,:], base_nodest[0,:]])
    base_link2 = np.vstack([base_nodes[1,:], base_nodest[1,:]])
    base_link3 = np.vstack([base_nodes[2,:], base_nodest[2,:]])
  
    ax.plot(
        base_plt[:,0],
        base_plt[:,1],
        base_plt[:,2],
        
        "r-",
        linewidth=1,
        label="Base",
    )
    ax.plot(
        base_link1[:,0],
        base_link1[:,1],
        base_link1[:,2],
        
        "b-",
        linewidth=1,
        label="Base",
    )
    ax.plot(
        
        base_link2[:,0],
        base_link2[:,1],
        base_link2[:,2],
        "b-",
        linewidth=1,
        label="Base",
    )
    ax.plot(
        
        base_link3[:,0],
        base_link3[:,1],
        base_link3[:,2],
        "b-",
        linewidth=1,
        label="Base",
    )
    ax.plot(
        base_plot[:, 0],
        base_plot[:, 1],
        base_plot[:, 2],
        "b-",
        linewidth=1,
        label="Base",
    )
    
    ax.plot(
        base_plot2[:, 0],
        base_plot2[:, 1],
        base_plot2[:, 2],
        "b-",
        linewidth=1,
        label="Base",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Stewart Platform 3D Model")
    ax.set_zlim(0, 1)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    #ax.legend()
    plt.show()
