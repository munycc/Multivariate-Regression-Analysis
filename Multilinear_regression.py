import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl  
mpl.use('qt5agg')


df = pd.read_csv ('project_data_20.09.csv', delimiter=',', header=None, skiprows=1, names=['Height', 'Lenght','Time', 'Weight'])
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.2)
pd.set_option('display.max_columns',None)
df.head ()
print (df)

#Multivariate regression analysis

X = df [['Height', 'Lenght', 'Weight']]
Y = df.Time

# Y = mX+c
# Initializing m and c with random numbers
k = X.shape[1] # return the total number of parameters in X 
m_current = np.zeros(k)
c_current=0
N = X.shape [0] # return the total number of values 
Y_current = 0
print (X)

# cost function
pre_cost = sum ((Y-Y_current)**2) /2

# learning_rate
learning_rate = (1 / pre_cost) 

#Ask the user for specific values for weight and height 
user_weight = int(input ('please enter the value of the total load you want to carry (in tons): '))
user_weight_np = np.array([user_weight]).reshape(1, -1)

user_height = int(input ('please enter the height of the structure(m): '))
user_height_np = np.array([user_height]).reshape(1, -1)

user_lenght = int(input ('please enter the lenght of the structure(m): '))
user_lenght_np = np.array([user_lenght]).reshape(1, -1)

df_input = pd.DataFrame({'height': [user_height], 'lenght': [user_lenght], 'weight': [user_weight]})
print (df_input.shape)
user_np = np.array([df_input]).reshape(1, -1)
print (user_np.shape)

# Update the gradient with respect to m and c values using f'(m) = m_grad and f'(c)= c_grad
# Repeating iterations to get lower cost function (error) and more precise Y
# To perform matrix multiplication, the first matrix must have the same number of columns as the second matrix has rows.
# That is why we are multiplying by transpose T
# Choose a number of iteration to test out the gradient, it can be lower or higher depending on the results, it can be changed

for i in range (25):
    
      m_grad = ((2/N) * (np.dot(X.T,(Y_current- Y))))
       
      c_grad = (2/N) * sum (Y_current-Y)


      # Multiply the gradient with the learning rate
      # Update m and c by substracting the derivative from initial numbers 
      
      m_update = m_current - m_grad * learning_rate
      c_update = c_current - c_grad * learning_rate


      #We need to sum all the X column values
      Y_update = (m_update * X). sum (axis = 1) + c_update
      
      

      #Cost function
      cost = sum ((Y_update - Y)**2) / N

      print ('cost', i, cost)

      #Verify that if the cost value gets lower, then we are on the right track

      if pre_cost > cost:
            m_current = m_update
            c_current = c_update
            pre_cost = cost
      
            # Trying to plot our values for only cost (to see how the cost influence the performance

            fig = plt.figure(figsize=(12,12))

            fig.subplots_adjust(left=0.1, bottom=0.1, right=1, top=0.9, wspace=0.4, hspace=0.4)
            
            # creating subplots 
            ax0 = plt.subplot2grid((3,6), (0,0), rowspan=1, colspan=2)
            ax1 = plt.subplot2grid((3,6), (1,0), rowspan=1, colspan=2)
            ax2 = plt.subplot2grid((3,6), (2,0), rowspan=1, colspan=2)
            ax3 = plt.subplot2grid((3,6), (0,2), rowspan=2, colspan=4, projection='3d')

        
            #sorting the values for weight before plotting
            sorted_x0 = np.sort(df.Weight, axis=None)
            sorted_x0 = sorted_x0[:, np.newaxis] # add a new axis
            sorted_y0 = (m_update * sorted_x0).sum(axis=1) + c_update

            #sorting the values for height before plotting
            sorted_x1 = np.sort(df.Height, axis=None)
            sorted_x1 = sorted_x1[:, np.newaxis] # add a new axis
            sorted_y1 = (m_update * sorted_x1).sum(axis=1) + c_update

            #sorting the values for lenght before plotting
            sorted_x2 = np.sort(df.Lenght, axis=None)
            sorted_x2 = sorted_x2[:, np.newaxis] # add a new axis
            sorted_y2 = (m_update * sorted_x2).sum(axis=1) + c_update
           

            ax0.scatter (df.Weight, df.Time, alpha = .2, color= 'red', label= 'actual')
            ax0.plot(sorted_x0, sorted_y0, 'b-', linewidth=2, label='fitted line')
            #ax0.scatter (df. Weight,(m_current *X).sum(axis=1) +c_current , alpha = .5, color= 'blue', label= 'fitted values')
            ax0.set_xlabel(r"$Weight(tons)$", fontsize=12)
            ax0.set_ylabel(r"$Time(hours)$", fontsize=12)
            ax0.plot(user_weight_np, (m_update *user_weight_np).sum(axis=1) + c_update , 'g*', markersize=15)

            
            ax1.scatter (df.Height, df.Time, alpha = .2, color= 'red', label= 'actual')
            ax1.plot(sorted_x1, sorted_y1, 'b-', linewidth=2, label='fitted line')
            #ax1.scatter (df.Height,(m_current *X).sum(axis=1) +c_current , alpha = .5, color= 'blue', label= 'fitted values')
            ax1.set_xlabel(r"$Height(m)$", fontsize=12)
            ax1.set_ylabel(r"$Time(hours)$", fontsize=12)
            ax1.plot(user_height_np, (m_update *user_height_np).sum(axis=1)+ c_update , 'g*', markersize=15)

            ax2.scatter (df.Lenght, df.Time, alpha = .2, color= 'red', label= 'actual')
            ax2.plot(sorted_x2, sorted_y2, 'b-', linewidth=2, label='fitted line')
            #ax1.scatter (df.Height,(m_current *X).sum(axis=1) +c_current , alpha = .5, color= 'blue', label= 'fitted values')
            ax2.set_xlabel(r"$Lenght(m)$", fontsize=12)
            ax2.set_ylabel(r"$Time(hours)$", fontsize=12)
            ax2.plot(user_lenght_np, (m_update *user_lenght_np).sum(axis=1)+ c_update , 'g*', markersize=15)

            
            #drawing 3d view

    
            ax3.scatter(df['Height'], df['Weight'], df['Time'], c='r', marker='o')
            ax3.set_xlabel('Height (m)')
            ax3.set_ylabel('Weight (tons)')
            ax3.set_zlabel('Time (h)')
            #Z= ax3.scatter(df['Height'], df['Weight'], Y_update, c='b', marker='o')
            ax3.scatter(user_height_np, user_weight_np, (m_update * user_np).sum(axis=1) + c_update, c='g', marker='*', s=150, label='prediction')

            
            # Add the following lines to create a mesh grid for the height and weight values
            # We will use this grid to plot the plane
            h = np.linspace(df.Height.min(), df.Height.max(), 50)
            w = np.linspace(df.Weight.min(), df.Weight.max(), 50)
            H, W = np.meshgrid(h, w)

            # Calculate the predicted time values for each height and weight combination
            Y_plane = m_current[0] * H + m_current[1] * W + c_current

            # Plot the plane
            ax3.plot_surface(H, W, Y_plane, alpha=0.2, cmap='viridis')
            

            plt.show ()
            fig.tight_layout()
            

 
      else:
            learning_rate = learning_rate / 10


print(cost,learning_rate)
print ("m_current =", m_current)
print ("c_current =", c_current)
print ("optimal time (h) =", (m_update * user_np).sum(axis=1) + c_update)


# calculate R2 (coefficient of determination) in order to find out how solid our model is
def r2(y_true, y_pred):
    y_bar = np.mean(y_true)
    ss_tot = np.sum((y_true - y_bar)**2)
    ss_res = np.sum((y_true - y_pred)**2)
    r2 = 1 - (ss_res/ss_tot)
    return r2


# calculate R2
Y_pred= (m_update * X).sum(axis=1) + c_update
r2_score = r2(Y,Y_pred)
print('R2 score: {:.4f}'.format(r2_score))

            





















