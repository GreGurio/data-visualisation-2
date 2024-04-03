import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import scipy.stats as stats



# Specify the path to your CSV file
file_path = 'data.csv'

# Use read_csv() function to import data from the CSV file
X = pd.read_csv(file_path)


# Згенеруємо уявні дані для прикладу (для 10 параметрів)
np.random.seed(0)
# X = 2 * np.random.rand(100, 10)  # Змінні X (100 спостережень, 10 параметрів)
coefficients = [0.5488135,  0.71518937, 0.602763318, 0.54488318, 0.4236548,  0.64589411, 0.43758721, 0.891773,   0.96366276, 0.38344152]  # Коефіцієнти для кожного параметру
noise = np.random.randn(100)  # Випадковий шум
y = np.dot(X, coefficients) + noise  # Залежна змінна y

# Побудова моделі регресії
model = LinearRegression()
model.fit(X, y)

# Знаходження прогнозованих значень
y_pred = model.predict(X)

# Виведення коефіцієнта детермінації
r_squared = r2_score(y, y_pred)
print("Коефіцієнт детермінації:", r_squared)

# Виведення коефіцієнта кореляції (можна також використовувати np.corrcoef())
corr_coef = np.corrcoef(y.T, y_pred.T)[0][1]
print("Коефіцієнт кореляції:", corr_coef)

m = 1

# Обрахунок F-критерію Фішера
Femp = r_squared / (1 - r_squared) * (100 - m - 1) / m
Femp2 = corr_coef / (1 - corr_coef) * (100 - m - 1) / m
print("F-критерій Фішера по коефіцієнту детермінації:", Femp)
print("F-критерій Фішера по коефіцієнту кореляції:", Femp2)



# # Побудова графіка
# plt.scatter(y, y_pred, color='blue')
# plt.plot(y, y, color='red', linestyle='--')  # Лінія ідентичності
# plt.xlabel('Справжні значення')
# plt.ylabel('Прогнозовані значення')
# plt.title('Парна лінійна регресія')
# plt.show()

#
# # Побудова графіка та довірчого інтервалу
plt.scatter(y, y_pred, color='blue')
plt.plot(y, y, color='red', linestyle='--')  # Лінія ідентичності

# Обчислення прогнозу середнього значення
mean_prediction = y_pred

# Визначення коефіцієнтів та кількості ступенів свободи для розподілу t
n = len(X)
p = 11  # Кількість параметрів моделі (включаючи інтерцепт)
t_value = stats.t.ppf(0.975, df=n - p - 1)

# Стандартне відхилення помилки
residuals = y - y_pred
std_error = np.sqrt(np.sum(residuals**2) / (n - p - 1))

# Довірчий інтервал для кожного значення X
confidence = t_value * std_error * np.sqrt(1 + 1 / n + (y - np.mean(y))**2 / np.sum((y - np.mean(y))**2))

# Верхня та нижня межі довірчого інтервалу
upper_bound = y_pred + confidence
lower_bound = y_pred - confidence

# Побудова довірчого інтервалу
plt.fill_between(y.flatten(), upper_bound.flatten(), lower_bound.flatten(), color='gray', alpha=0.2, label='Довірчий інтервал')

plt.xlabel('Справжні значення')
plt.ylabel('Прогнозовані значення')
plt.title('Лінійна парна регресія з довірчим інтервалом')
plt.legend()
plt.show()