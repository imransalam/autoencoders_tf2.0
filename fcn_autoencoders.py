import tensorflow as tf
tf.enable_eager_execution()


class FullyConnectedAutoEncoder(tf.keras.Model):
    def __init__(self):
        super(FullyConnectedAutoEncoder, self).__init__()
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        
        
        self.bottleneck = tf.keras.layers.Dense(16, activation=tf.nn.relu)
    
        self.dense4 = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.dense5 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        
        self.dense_final = tf.keras.layers.Dense(784)
        
    
    def call(self, inp):
        x_reshaped = self.flatten_layer(inp)
        x = self.dense1(x_reshaped)
        x = self.dense2(x)
        x = self.bottleneck(x)
        x = self.dense4(x)
        x = self.dense5(x)
        x = self.dense_final(x)
        return x, x_reshaped

def loss(x, x_bar):
    return tf.losses.mean_squared_error(x, x_bar)

def grad(model, inputs):
    with tf.GradientTape() as tape:
        reconstruction, inputs_reshaped = model(inputs)
        loss_value = loss(inputs_reshaped, reconstruction)
    return loss_value, tape.gradient(loss_value, model.trainable_variables), inputs_reshaped, reconstruction


if __name__ == '__main__':

	(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
	x_train = x_train.astype('float32') / 255.
	x_test = x_test.astype('float32') / 255.
	
	model = FullyConnectedAutoEncoder()
	optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
	global_step = tf.Variable(0)

	num_epochs = 50
	batch_size = 4


	for epoch in range(num_epochs):
	    print("Epoch: ", epoch)
	    for x in range(0, len(x_train), batch_size):
	        x_inp = x_train[x : x + batch_size]
	        loss_value, grads, inputs_reshaped, reconstruction = grad(model, x_inp)
	        optimizer.apply_gradients(zip(grads, model.trainable_variables),
	                              global_step)
	        
	        if global_step.numpy() % 200 == 0:
	            print("Step: {},         Loss: {}".format(global_step.numpy(),
	                                          loss(inputs_reshaped, reconstruction).numpy()))


