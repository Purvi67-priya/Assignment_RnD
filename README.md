📘 Parametric Curve Modeling and Optimization
🧮 Problem Overview

We are given the following parametric equation of a curve:

𝑥
(
𝑡
)
	
=
𝑡
cos
⁡
(
𝜃
)
−
𝑒
𝑀
∣
𝑡
∣
sin
⁡
(
0.3
𝑡
)
sin
⁡
(
𝜃
)
+
𝑋


𝑦
(
𝑡
)
	
=
42
+
𝑡
sin
⁡
(
𝜃
)
+
𝑒
𝑀
∣
𝑡
∣
sin
⁡
(
0.3
𝑡
)
cos
⁡
(
𝜃
)
x(t)
y(t)
	​

=tcos(θ)−e
M∣t∣
sin(0.3t)sin(θ)+X
=42+tsin(θ)+e
M∣t∣
sin(0.3t)cos(θ)
	​


Our objective is to determine the parameters

𝜃
θ, 
𝑀
M, and 
𝑋
X that best fit a given set of data points 
(
𝑥
𝑖
,
𝑦
𝑖
)
(x
i
	​

,y
i
	​

) for the interval 
6
<
𝑡
<
60
6<t<60.

⚙️ Methodology
1️⃣ Data Loading

The dataset (xy_data.csv) contains observed 
(
𝑥
,
𝑦
)
(x,y) coordinates corresponding to evenly spaced 
𝑡
t-values within 
6
≤
𝑡
≤
60
6≤t≤60.

Each point represents a sample along the unknown curve.

2️⃣ Model Formulation

The mathematical model is expressed as:

𝑥
𝑝
𝑟
𝑒
𝑑
(
𝑡
)
	
=
𝑡
cos
⁡
(
𝜃
)
−
𝑒
𝑀
∣
𝑡
∣
sin
⁡
(
0.3
𝑡
)
sin
⁡
(
𝜃
)
+
𝑋


𝑦
𝑝
𝑟
𝑒
𝑑
(
𝑡
)
	
=
42
+
𝑡
sin
⁡
(
𝜃
)
+
𝑒
𝑀
∣
𝑡
∣
sin
⁡
(
0.3
𝑡
)
cos
⁡
(
𝜃
)
x
pred
	​

(t)
y
pred
	​

(t)
	​

=tcos(θ)−e
M∣t∣
sin(0.3t)sin(θ)+X
=42+tsin(θ)+e
M∣t∣
sin(0.3t)cos(θ)
	​


These equations generate predicted coordinates 
(
𝑥
𝑝
𝑟
𝑒
𝑑
,
𝑦
𝑝
𝑟
𝑒
𝑑
)
(x
pred
	​

,y
pred
	​

) for any given parameter set.

3️⃣ Objective Function

To evaluate the model’s accuracy, we minimize the mean Euclidean distance between observed and predicted points:

𝐽
(
𝜃
,
𝑀
,
𝑋
)
=
1
𝑁
∑
𝑖
=
1
𝑁
(
𝑥
𝑖
−
𝑥
𝑝
𝑟
𝑒
𝑑
,
𝑖
)
2
+
(
𝑦
𝑖
−
𝑦
𝑝
𝑟
𝑒
𝑑
,
𝑖
)
2
J(θ,M,X)=
N
1
	​

i=1
∑
N
	​

(x
i
	​

−x
pred,i
	​

)
2
+(y
i
	​

−y
pred,i
	​

)
2
	​


This cost function ensures the optimized parameters yield the curve closest to the observed data.

4️⃣ Optimization Setup
Parameter	Range

𝜃
θ	[0°, 50°]

𝑀
M	[-0.05, 0.05]

𝑋
X	[0, 100]

Algorithm: L-BFGS-B (for bounded optimization)

Initial Guess: [25°, 0, 50]

Library Used: scipy.optimize.minimize

🧩 Estimated Best-Fit Parameters
Parameter	Optimized Value

𝜃
θ	30.0441°

𝑀
M	−0.00528

𝑋
X	55.3473
🧠 Defining Parameter 
𝑡
𝑖
t
i
	​


Since the data file includes only 
(
𝑥
𝑖
,
𝑦
𝑖
)
(x
i
	​

,y
i
	​

),
the corresponding 
𝑡
𝑖
t
i
	​

 values are reconstructed using uniform spacing over 
[
6
,
60
]
[6,60].

𝑡
𝑖
=
6
+
(
𝑖
−
1
)
(
60
−
6
)
𝑁
−
1
t
i
	​

=6+
N−1
(i−1)(60−6)
	​

	​


This ensures:

𝑡
1
=
6
t
1
	​

=6

𝑡
𝑁
=
60
t
N
	​

=60

All 
𝑡
𝑖
t
i
	​

 are equally spaced in between.

In Python:

t = np.linspace(6, 60, N)

💻 Python Implementation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load the dataset
data = pd.read_csv("xy_data.csv")
x_obs, y_obs = data.iloc[:, 0].values, data.iloc[:, 1].values
t = np.linspace(6, 60, len(x_obs))

# Define the parametric model
def model(params, t):
    theta_deg, M, X = params
    theta = np.deg2rad(theta_deg)
    exp_term = np.exp(M * np.abs(t))
    x_pred = t*np.cos(theta) - exp_term*np.sin(0.3*t)*np.sin(theta) + X
    y_pred = 42 + t*np.sin(theta) + exp_term*np.sin(0.3*t)*np.cos(theta)
    return x_pred, y_pred

# Define the objective function
def objective(params):
    x_pred, y_pred = model(params, t)
    return np.mean(np.sqrt((x_obs - x_pred)**2 + (y_obs - y_pred)**2))

# Run optimization
bounds = [(0, 50), (-0.05, 0.05), (0, 100)]
res = minimize(objective, [25, 0, 50], bounds=bounds, method='L-BFGS-B')

theta, M, X = res.x
print(f"Theta = {theta:.6f}°, M = {M:.6f}, X = {X:.6f}")

# Plot the observed data vs fitted curve
x_fit, y_fit = model(res.x, t)
plt.scatter(x_obs, y_obs, s=10, alpha=0.6, label='Observed Points')
plt.plot(x_fit, y_fit, 'r', lw=2, label='Fitted Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.title('Observed Data vs Fitted Parametric Curve')
plt.show()

📈 Results & Visualization

The resulting fitted curve (shown in red) closely follows the observed data points (in blue),
indicating a highly accurate parameter estimation.

🔍 Step-by-Step Summary

Problem Understanding: Estimate 
𝜃
,
𝑀
,
𝑋
θ,M,X from given 
(
𝑥
,
𝑦
)
(x,y) data.

Data Loading: Import points from xy_data.csv.

Parameter Mapping: Generate uniform 
𝑡
𝑖
t
i
	​

 values in [6, 60].

Model Definition: Encode the parametric equations.

Loss Function: Compute mean Euclidean distance.

Optimization: Apply scipy.optimize.minimize with bounds.

Visualization: Plot fitted vs. observed data.

Result Interpretation: Analyze optimized parameters.

Documentation: Prepare clear README for reproducibility.

✅ Conclusion

The optimized curve accurately represents the data distribution.

The small negative 
𝑀
M value introduces a damping effect, slightly reducing amplitude for larger 
∣
𝑡
∣
∣t∣.

Final parameters:

𝜃
=
30.04
°
θ=30.04°, 
𝑀
=
−
0.00528
M=−0.00528, 
𝑋
=
55.35
X=55.35

Demonstrates a complete parametric curve-fitting workflow using Python and SciPy — from data preprocessing to optimization and visualization
