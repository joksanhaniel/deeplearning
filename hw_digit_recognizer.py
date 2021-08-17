import pandas as pd
import numpy as np
#import sklearn
#import sklearn.datasets
#import sklearn.linear_model
#from sklearn.model_selection import train_test_split
#import sklearn.preprocessing
import pickle


class HWDigitRecognizer:

  def __init__(self, train_filename, test_filename):

    """ 
    El método init leerá los datasets con extensión ".csv" cuyas ubicaciones son  
    recibidas mediante los paramentros <train_filename> y <test_filename>. 
    Los usará para crear las matrices de X_train, X_test, Y_train y Y_test 
    con las dimensiones adecuadas y normalización de acuerdo a lo definido 
    en clase para un problema de clasificación multiclase resuelto media una 
    única red neuronal. """
    
    #self.train_filename= train_filename
    #self.test_filename=test_filename

    #train_filename='./autograder_data/mnist_train_0.01sampled.csv'
    #test_filename='./autograder_data/mnist_test_0.01sampled.csv'


    #train_filename='./datasets/mnist_train.csv'
    #test_filename='./datasets/mnist_test.csv'

    train=np.array(pd.read_csv(train_filename,header=0))
    test=np.array(pd.read_csv(test_filename, header=0))
    all_data=np.transpose(train)
    all_data_prueba=np.transpose(test)
    original_y=np.array([all_data[0]])
    original_yprueba=np.array([all_data_prueba[0]])

    #TECNICA OHE
    def one_hot_encode(joksan):
      m=joksan.shape[1]
      elementos=np.zeros((10,m))
      for i in range(0,m):
        x=joksan[0][i]
        for j in range(0,10):
          if j==x:
            elementos[x][i]=1  
      return elementos  
    Y_train=one_hot_encode(original_y)
    Y_test=one_hot_encode(original_yprueba)
    original_x_train=all_data[1:]

    #NORMALIZE
    X_train=original_x_train/255
    original_x_test=all_data_prueba[1:]
    X_test=original_x_test/255
    
    self.X_train=X_train
    self.Y_train=Y_train
    self.X_test=X_test
    self.Y_test=Y_test
  
#data=HWDigitRecognizer(train_filename=np.array(pd.read_csv('./autograder_data/mnist_train_0.01sampled.csv')),test_filename=np.array(pd.read_csv('./autograder_data/mnist_test_0.01sampled.csv')))

#print(data.train_filename,data.test_filename)

  def train_model(self):
    X=self.get_datasets()["X_train"]
    Y=self.get_datasets()["Y_train"]
    print_cost=True

    learning_rate=0.1 
    num_iterations=20000
    layers_dims=[X.shape[0],10,5,10]

    #learning_rate=0.1                 #99.88 prueba
    #num_iterations=20000
    #layers_dims=[X.shape[0],10,5,10]

  #DATOS USADOS PARA EL DATASET PEQUEÑO
    #learning_rate=0.5 
    #num_iterations=1500
    #layers_dims=[X.shape[0],43,30,35,10]
    
    #learning_rate=0.6
    #num_iterations=3000
    #layers_dims=[X.shape[0],85,60,70,10]

    
    """
    Entrena complementamente una red neuronal con múltiples capas, utilizando la función de activación RELU en las primeras L-1 capas y la función Softmax en la última capa de tamaño 10
    para realizar clasificación multiclase. 

    Retorna una tupla cuyo primer elemento es un diccionario que contiene los parámetros W y b de todas las capas del modelo con esta estructura:

    { "W1": ?, "b1": ?, ... , "WL": ?, "bL": ?}

    donde los signos <?> son sustituidos por arreglos de numpy con los valores respectivos. El valor de L será elegido por el estudiante mediante
    experimentación. El segundo elemento a retornar es una lista con los costos obtenidos durante el entrenamiento cada 100 iteraciones.

    Por razones de eficiencia el autograder revisará su programa usando un dataset más pequeño que el que se proporciona (usted puede hacer lo mismo para sus pruebas iniciales). Pero una vez entregado su proyecto se harán pruebas con el dataset completo, por lo que el diccionario que retorna este método con los resultados del entrenamiento con una precisión mayor al 95% en los datos de prueba debe ser entregado junto con este archivo completado.
    
    Para entregar dicho diccionario deberá guardarlo como un archivo usando el módulo "pickle" con el nombre y extensión "params.dict", este archivo deberá estar ubicado en el mismo directorio donde se encuentra  el archivo actual: hw_digit_recognizer.py. El autograder validará que este archivo esté presente y tendra las claves correctas, pero la revisión de la precisión se hará por el docente después de la entrega. La estructura del archivo será la siguiente:
    
    {
      "model_params": { "W1": ?, "b1": ?, ... , "WL": ?, "bL": ?},
      "layer_dims": [30, ..., 10],  
      "learning_rate": 0.001,
      "num_iterations": 1500,
      "costs": [0.2356, 0.1945, ... 0.00345]
    }

    <model_params>, es un diccionario que contiene los valores de las L * 2  matrices de parámetros del modelo (numpy array) desde W1 y b1, hasta WL y bL, el valor de L será elegido por el estudiante.
    <layer_dims>, es una lista con las dimensiones de las L+1 capas del modelo, incluyendo la capa de entrada.
    <learning_rate>, el valor del ritmo de entrenamiento.
    <num_iterations>, el número de iteraciones que se usó.
    <costs>, una lista con los costos obtenidos cada 100 iteraciones.
    """
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters_deep(layers_dims)
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 100 == 0:
            costs.append(cost)  

    diccionary={
      "model_params": parameters,
      "layer_dims": layers_dims,  
      "learning_rate": learning_rate,
      "num_iterations": num_iterations,
      "costs": costs
    }

    archivo=open('params.dict','wb')

    pickle.dump(diccionary, archivo) 

    archivo.close()

    return parameters,costs


  def predict(self, X, model_params):
    """
    Retorna una matriz de predicciones de <(1,m)> con valores entre 0 y 9 que representan las etiquetas para el dataset X de tamaño <(n,m)>.

    <model_params> contiene un diccionario con los parámetros <w> y <b> de cada uno de los clasificadores tal como se explica en la documentación del método <train_model>.
    """

    AL, caches =L_model_forward(X,model_params)

    Y_gorrito=np.max(AL,axis=0,keepdims=True)

    #tecnica OHE
    OHE=np.zeros((AL.shape[0],AL.shape[1]))
    Matriz=np.zeros((1,AL.shape[1]))
    for i in range(0,AL.shape[1]):
      for j in range(0,AL.shape[0]):
        if AL[j][i]==Y_gorrito[0][i]:
          OHE[j][i]=1
    
    for i in range(0,OHE.shape[1]):
      for j in range(0,OHE.shape[0]):
        if OHE[j][i]==1:
          Matriz[0][i]=j
    return Matriz

  def get_datasets(self):
    """Retorna un diccionario con los datasets preprocesados con los datos y 
    dimensiones que se usaron para el entrenamiento
    
    d = { "X_train": X_train,
    "X_test": X_test,
    "Y_train": Y_train,
    "Y_test": Y_test
    }
    """

    d = { "X_train": self.X_train,
    "X_test": self.X_test,
    "Y_train": self.Y_train,
    "Y_test": self.Y_test
    }

    return d

#_____________FUNCIONES EXTRA, EXTRAIDAS DE LOS LABORATORIOS____________________________

#PROPAGACION HACIA ADELANTE

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    return parameters


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    if activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        e=np.exp(Z-np.max(Z))
        softmax= e/e.sum(axis=0, keepdims=True)
        A=softmax 
        activation_cache = Z
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A=np.maximum(0,Z)
        activation_cache = Z
        
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
  
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A 
        W=parameters["W"+str(l)]
        b=parameters["b"+str(l)]
        A, cache = linear_activation_forward(A_prev, W, b, activation='relu')
        caches.append(cache)
    W=parameters["W"+str(L)]
    b=parameters["b"+str(L)]
    AL, cache =linear_activation_forward(A, W, b, activation="softmax")
    caches.append(cache)
    assert(AL.shape == (10,X.shape[1]))    
    return AL, caches

def compute_cost(AL, Y): 
    m = Y.shape[1]
    cost=np.mean(Y*np.log((AL + 1*(np.exp(-8)))))
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1/m*dZ.dot(A_prev.T)
    db = 1/m*np.sum(dZ, axis=1, keepdims=True)
    dA_prev = (W.T).dot(dZ)
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db


#PROPAGACION HACIA ATRAS

def linear_activation_backward(dA, cache, activation,AL,Y):
    linear_cache, activation_cache = cache
    if activation == "relu":
        Z=activation_cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        assert (dZ.shape == Z.shape)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)     
    elif activation == "softmax":
        dZ = AL-Y
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
   
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL= np.sum(Y/AL)
   
    current_cache = caches[-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, 'softmax',AL,Y)
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l+1)],current_cache,'relu',AL,Y)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 

    for l in range(L):
        parameters["W" + str(l+1)] -=learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l+1)] -=learning_rate * grads["db" + str(l + 1)]
   
    return parameters