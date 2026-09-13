# Note 01: Data Visualization Techniques
# ======================================

# Import Libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import seaborn as sns

# 01. Scatter Plot
# ================

x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 8, 10]

plt.figure()
plt.scatter(x, y, s=30, c="purple", marker="o")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter Plot")
plt.grid(True, alpha=0.3)
plt.show()

# 02. Bar Plot
# ============

# 1. Simple Bar Chart - Vertical Chart
products = ["A", "B", "C", "D"]
sales = [100, 150, 80, 200]

plt.figure()
plt.bar(products, sales, width=0.4, color="green")
plt.xlabel("Products")
plt.ylabel("Sales")
plt.title("Bar Chart")
plt.grid(True, alpha=0.3)
plt.show()

# 2. Simple Bar Chart - Horizontal Chart
products = ["A", "B", "C", "D"]
sales = [100, 150, 80, 200]

plt.figure()
plt.barh(products, sales, height=0.4, color="yellow")
plt.ylabel("Products")
plt.xlabel("Sales")
plt.title("Bar Chart")
plt.grid(True, alpha=0.3)
plt.show()

# 3. Multiple Bar Chart - Vertical Chart
cats = ['A', 'B', 'C', 'D']
v1, v2 = [4, 7, 1, 8], [5, 6, 2, 9]
w, x = 0.4, np.arange(len(cats))

plt.figure()
plt.bar(x - w/2, v1, w, label='Set 1')
plt.bar(x + w/2, v2, w, label='Set 2')
plt.xticks(x, cats)
plt.xlabel('Cats')
plt.ylabel('Values')
plt.title('Grouped Bar Chart')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 4. Multiple Bar Chart - Horizontal Chart
cats = ['A', 'B', 'C', 'D']
v1, v2, v3 = [4, 7, 1, 8], [5, 6, 2, 9], [9, 10, 11, 13]
w, x = 0.3, np.arange(len(cats))

plt.figure()
plt.barh(x - w, v1, height=w, label='Set 1')
plt.barh(x, v2, height=w, label='Set 2')
plt.barh(x + w, v3, height=w, label='Set 3')
plt.yticks(x, cats)
plt.xlabel('Values')
plt.ylabel('Cats')
plt.title('Grouped Bar Chart')
plt.legend()
plt.grid(True, alpha=0.3, linestyle='--')
plt.show()

# 5. Stacked Bar Chart - Vertical Chart
x = ['A', 'B', 'C', 'D']
y1 = [10, 20, 10, 30]
y2 = [20, 25, 15, 25]

plt.figure()
plt.bar(x, y1, color='red')
plt.bar(x, y2, bottom=y1, color='b')
plt.grid(True, alpha=0.3)
plt.ylabel("Values")
plt.title("Bar Chart")
plt.xlabel("Cats")
plt.show()

# 6. Stacked Bar Chart - Horizontal Chart
y = ['A', 'B', 'C', 'D']
x1 = [10, 20, 10, 30]
x2 = [20, 25, 15, 25]

plt.figure()
plt.barh(y, x1, color='purple')
plt.barh(y, x2, left=x1, color='blue')
plt.grid(True, alpha=0.3)
plt.ylabel("Cats")
plt.title("Bar Chart")
plt.xlabel("Values")
plt.show()

# 03. Pie Chart
# =============

labels = ["IT", "Business", "Engineering", "Arts"]
values = [40, 25, 20, 15]

plt.figure()
plt.pie(values, labels=labels, autopct="%1.2f%%", wedgeprops={"edgecolor": "black", "linewidth": 1})
plt.title("Student Distribution")
plt.show()

# 04. Line Chart
# ==============

months = ["Jan", "Feb", "Mar", "Apr", "May"]
sales = [100, 120, 150, 140, 180]

plt.figure()
plt.plot(months, sales, linestyle="--", color="red")
plt.scatter(months, sales, s=30, c="purple", marker="o", label="2021")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Monthly Sales")
plt.grid(True, alpha=0.3)
plt.legend(loc="lower right")
plt.show()

# Legend Locations
# ================

# +--------------+--------------+---------------+
# | 'upper left' |'upper center'| 'upper right' |
# +--------------+--------------+---------------+
# |'center left' |   'center'   |'center right' |
# +--------------+--------------+---------------+
# | 'lower left' |'lower center'| 'lower right' |
# +--------------+--------------+---------------+

# 05. Histogram
# =============

marks = [
    55, 60, 62, 65, 65,
    68, 68, 70, 70, 70,
    72, 72, 72, 72, 73,
    73, 74, 74, 75, 75,
    75, 76, 76, 76, 77,
    77, 78, 78, 78, 80,
    80, 81, 82, 82, 83,
    84, 85, 86, 88, 90
]

# Normal Distribution Curve
mean = np.mean(marks)
std = np.std(marks)
x = np.linspace(min(marks), max(marks), 100)
y = norm.pdf(x, mean, std)

# Plot Graph and Curve
plt.figure()
plt.hist(marks, bins=10, density=True, edgecolor="black", linewidth=1.5)
plt.plot(x, y, linewidth=2, linestyle="--", color="red")
plt.xlabel("Marks")
plt.ylabel("Density")
plt.title("Distribution of Marks")
plt.grid(True, alpha=0.3)
plt.show()

# 06. Boxplot
# ===========

marks_A = [45, 50, 52, 55, 60, 62, 65, 65, 70, 72, 75, 80, 85, 95]
marks_B = [50, 55, 58, 60, 63, 65, 68, 70, 72, 75, 78, 80, 82, 85]
marks_C = [40, 45, 48, 50, 55, 58, 60, 62, 65, 68, 70, 72, 75, 78]

plt.figure()
plt.boxplot([marks_A, marks_B, marks_C])
plt.xticks([1, 2, 3], ["Class A", "Class B", "Class C"])
plt.ylabel("Marks")
plt.xlabel("Classes")
plt.title("Distribution of Marks")
plt.grid(True, alpha=0.3, linestyle='--')
plt.show()

# 07. Heatmap
# ===========

data = np.random.rand(10, 10) * 100

plt.figure()
sns.heatmap(
    data,
    xticklabels=list("ABCDEFGHIJ"),
    yticklabels=False,
    cmap="coolwarm",
    linewidths=0.5,
    annot=True,
    fmt=".1f"
)
plt.title("Heatmap", fontsize=16)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# 08. Area Chart
# ==============

months = ["Jan", "Feb", "Mar", "Apr", "May"]
sales_2025 = [100, 120, 150, 140, 180]
sales_2026 = [110, 130, 160, 155, 200]

plt.figure()
plt.fill_between(months, sales_2025, sales_2026, alpha=0.3)
# plt.fill_between(months, sales_2026, alpha=0.3)
plt.plot(months, sales_2025, marker="o", label="2025")
plt.plot(months, sales_2026, marker="o", label="2026")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Sales Over Time")
plt.legend()
plt.grid(True, alpha=0.3, linestyle='--')
plt.show()

# 09. 3D Plot
# ===========

# 1. 3D Scatter Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
# 111 => 1 Row / 1 Column / 1 Plot
# 3d => Prepare a 3D Canvas
x = [1, 2, 3, 4, 5]
y = [2, 4, 1, 5, 3]
z = [5, 3, 4, 2, 6]

ax.scatter(x, y, z)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title("3D Scatter Plot")
ax.grid(True, alpha=0.3)
plt.show()

# 2. 3D Line Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
x = [1, 2, 3, 4, 5]
y = [2, 4, 1, 5, 3]
z = [5, 3, 4, 2, 6]

ax.plot(x, y, z)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title("3D Line Plot")
plt.show()

# 3. 3D Surface Plot
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
# Meshgrid combines every x with every y
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title("3D Surface Plot")
plt.show()

# 10. Error Bar Plot
# ==================

# Data
subjects = ['A', 'B', 'C', 'D', 'E']

# Mean values
mean = np.array([70, 75, 65, 80, 72])

# Error values
error = np.array([5, 4, 6, 3, 5])

# Create error bar plot
plt.errorbar(
    subjects,
    mean,
    yerr=error, # Vertical Error
    # xerr=error, # Horizontal Error
    fmt='o',
    capsize=5,
    markersize=7
)

# Labels and title
plt.xlabel("Subjects")
plt.ylabel("Mean Marks")
plt.title("Mean Marks with Error Bars")

# Grid
plt.grid(True)

# Display plot
plt.show()