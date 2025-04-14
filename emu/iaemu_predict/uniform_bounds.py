import numpy as np

log_M_min = np.array([11.35, 11.46, 11.60, 11.75, 12.02, 12.30, 12.79, 13.38, 14.22])
sigma_log_M = np.array([0.25, 0.24, 0.26, 0.28, 0.26, 0.21, 0.39, 0.51, 0.77])
log_M_zero = np.array([11.20, 10.59, 11.49, 11.69, 11.38, 11.84, 11.92, 13.94, 14.00])
log_M_one = np.array([12.40, 12.68, 12.83, 13.01, 13.31, 13.58, 13.94, 13.91, 14.69])
alpha = np.array([0.83, 0.97, 1.02, 1.06, 1.06, 1.12, 1.15, 1.04, 0.87])

linear_fit = np.polyfit(log_M_min, log_M_zero, 1)
rmse = 4*np.sqrt(np.mean((np.polyval(linear_fit, log_M_min) - log_M_zero)**2))
print("RMSE of linear fit, log_M_min and log_M_zero: {:.2f}".format(rmse))

linear_fit = np.polyfit(log_M_min, log_M_one, 1)
rmse = 4*np.sqrt(np.mean((np.polyval(linear_fit, log_M_min) - log_M_one)**2))
print("RMSE of linear fit, log_M_min and log_M_one: {:.2f}".format(rmse))

linear_fit = np.polyfit(log_M_min, alpha, 1)
rmse = 4*np.sqrt(np.mean((np.polyval(linear_fit, log_M_min) - alpha)**2))
print("RMSE of linear fit, log_M_min and alpha: {:.2f}".format(rmse))

linear_fit = np.polyfit(log_M_min, sigma_log_M, 1)
rmse = 4*np.sqrt(np.mean((np.polyval(linear_fit, log_M_min) - sigma_log_M)**2))
print("RMSE of linear fit, log_M_min and sigma_log_M: {:.2f}".format(rmse))



