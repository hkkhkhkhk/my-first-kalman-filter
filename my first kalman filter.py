import numpy as np
import matplotlib.pyplot as plt

Estimation_error = 0 # dynamic and looped
Measurement_variance = 100 #static, how confident you are in your sensors (in n*Var)
Proccess_noise_variance = 3 #Static, how confident you are in your model (in n*Var)

Model_error = 1/80 # this is how much our model guesses wrong, inject some constant to make it drift
#you shouldn't be able to change the model error in real life as it is limited by how rubbish your model is

def Kalman_gain(Eest, Emea):
    return Eest / (Eest + Emea)

def state_estimator(current, step_num):
    return current + time_step*np.cos(x_values[step_num]) + Model_error # introduce some consistent error to simulate integration error of a model

def corrected_estimate(prev_state_estimate, measurement, k):
    return prev_state_estimate + k*(measurement - prev_state_estimate)

def update_estimation_error(prev_estimation_error, k):
    return (1 - k) * prev_estimation_error + Proccess_noise_variance

# Generate data using a loop
num_steps = 100
x_values = np.linspace(0, 2 * np.pi, num_steps)
time_step = 2 * np.pi/num_steps

true_sine_wave = np.zeros(num_steps)
noisy_sine_wave = np.zeros(num_steps)
state_prediction = np.zeros(num_steps)
corrected_state = np.zeros(num_steps)


for i in range(num_steps):
    true_sine_wave[i] = np.sin(x_values[i])
    noisy_sine_wave[i] = true_sine_wave[i] + np.random.normal(0, 0.1)

    K = Kalman_gain(Estimation_error, Measurement_variance)
    # State prediction that is not corrected, will go into the cosmos
    state_prediction[i] = state_estimator(state_prediction[i-1], i)
    
    #state prediction correced by the noisy current signal
    corrected_state[i] = corrected_estimate(state_estimator(corrected_state[i-1], i), noisy_sine_wave[i], K)

    #lastly lets update the estimation error
    Estimation_error = update_estimation_error(Estimation_error, K)

# Plot the sine wave
plt.plot(x_values, true_sine_wave, label='True Sine Wave', color='blue')

# Plot the noisy sine wave
plt.plot(x_values, noisy_sine_wave, label='Noisy Sine Wave', color='red')

# Plot the state prediction
plt.plot(x_values, state_prediction, label='State Prediction (Raw)', color='green', linestyle='dashed')

plt.plot(x_values, corrected_state, label='State Prediction after Kalman (Corrected)', color='black')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('True Sine Wave, Noisy Measurement, and State Prediction')

plt.rcParams['figure.dpi'] = 800
# Add a legend
plt.legend()

# Show the plot
plt.show()
