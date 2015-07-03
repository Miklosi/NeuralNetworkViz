//Code adapted from 
//https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py
	
var NeuralNetwork = function(sizes) { 

	var self = this;

	//sizes: no. neurons at each layer
	//e.g. [2,2,1] = [2 (input), 2 (hidden layer1), 1 (output)]
	//for truth-tables: [2,2,1]
	//for character-recognition: [784, 30, 10]
	NeuralNetwork.prototype.init = function(sizes) {

		self.num_layers = sizes.length;
		self.sizes = sizes;

		var initialiseFn = function() { 
								return (Math.random() -.5) * 2/Math.sqrt(self.sizes[0]); 
							};

		var matrixApply = function(m, FUN) {
			return m.map(function(v) { 
							return v.map(function(w) { return FUN() }) 
						});
		};


		self.biases =   self.sizes.slice(1, self.sizes.length)
		                    .map(function(x, i) { 
		                   	  return math.matrix(matrixApply(math.ones([x, 1]), initialiseFn));
		                    });
		self.weights = self.sizes.slice(1, self.sizes.length)
		                   .map(function(x, i) { 
								return math.matrix(matrixApply(math.ones([x, self.sizes[i]]), initialiseFn));
		                    });

/*
		//TEST STATIC DATA
		self.biases = [
		    math.matrix([[0.262628827476874], [0.0737376578617841]])
		    , 
		    math.matrix([[0.4326041475869715]])
		];
		self.weights = [
		    math.matrix([
		    			[0.8140114056877792, 0.7331433838699013], 
						[0.7208098769187927, 0.12974167615175247]]),
		    math.matrix([[0.23425847105681896, 0.6160134135279804]])
		];

		var e = 1;*/

	};

	//training_data: [{ value: x, target: y },...]
	//eta: learning rate (0.25)
	NeuralNetwork.prototype.train = function(training_data, epochs, mini_batch_size, eta, test_data) {//,test_data=None
		SGD(training_data, epochs, mini_batch_size, eta, test_data);
	};

	//*
	NeuralNetwork.prototype.evaluate = function(test_data) {
		//return (for test_data) the no. cases where self.feed_forward(inputs) == test_data.outputs

		return d3.sum(test_data.map(function(d, i) {

			//tentative evaultion for character recognition:
//			var argmax = feedForward(d)
//							.map(function (output, i) {
// 								return { value: output, digit: 0 }
// 							})
// 							.sort(function (a, b) { 
// 								return b.value - a.value;
// 							})[0].digit;
//			return d.target[0][0] == argmax ? 1 : 0;

			//evaluation for AND/OR/etc. logic gates:
			var target = d.target[0][0],
				actual = feedForward(d.value)._data[0][0];

			//console.log("input: %s, output: %f, target: %f", math.format(d.value), actual, target);

			return Math.round(actual.toFixed(3)) == Math.round(target.toFixed(3)) ? 1 : 0;
 		}));
	};

	var SGD = function(training_data, epochs, mini_batch_size, eta, lambda, test_data) {
		var n = training_data.length;
		var lambda = lambda || 0;

		for (var j = 1; j <= epochs; j++) {
			var epoch_training_data = training_data; //d3.shuffle(training_data);

		  	var mini_batches = [];
		  	for (var k=0; k < n; k += mini_batch_size) {
		    	mini_batches.push(epoch_training_data.slice(k, k+mini_batch_size));
		  	}

		  	for (var m = 0; m < mini_batches.length; m++) {
		    	update_mini_batch(mini_batches[m], eta, lambda, training_data.length);
		  	}

		  	if(test_data)
		  		console.log("Epoch %d: %f / %f", j, self.evaluate(test_data), n);
		  	else
		  		console.log("Epoch %d complete", j);
		}
	};

	var feedForward = function(a) { //a: inputs, eg.: [[1],[1]]
		for (var i = 0; i < self.weights.length; i++) {
		  a = sigmoid_vec(
				math.chain(self.weights[i])
					.multiply(a)
					.add(self.biases[i]).done()
		      ); 
		}
		return a;
	};

	var backprop = function(x, y) {

		var nabla_b = zeroMatrixCopyOf(self.biases);
		var nabla_w = zeroMatrixCopyOf(self.weights);

		//feed-foward: get activations & component z-scores (weighted sum of inputs)
		var activation = x,
		    activations = [x],
		    zs = [];

		for (var k = 0; k < self.biases.length; k++) {
		    var w = self.weights[k], 
		        b = self.biases[k];
		    var z = math.chain(w)
		    			.multiply(activation)
		    			.add(b).done();

		    zs.push(z);
		    activations.push(sigmoid_vec(z));
		}

		//output error: (via cost derivative), how inaccurate are activations vs. targets?
		var delta = math.chain(cost_derivative(activations[activations.length-1], y))
						.dotMultiply(sigmoid_prime_vec(zs[zs.length-1])).done();

		nabla_b[nabla_b.length-1] = delta;
		nabla_w[nabla_w.length-1] = math.chain(delta)
										.multiply(
											math.chain(activations[activations.length-2])
												.transpose().done()
										).done();

		//backpropagate error: determine output error contributions from each previous layer
		for (var l = 2; l < self.num_layers; l++) {
			var z = zs[zs.length-l];
			var spv = sigmoid_prime_vec(z);

			//error at layer 'l'
			delta = math.chain(self.weights[self.weights.length-(l-1)])
						.transpose()
						.multiply(delta)
						.dotMultiply(spv).done();

			nabla_b[nabla_b.length-l] = delta; 
			nabla_w[nabla_w.length-l] = math.chain(delta)
											.multiply(
												math.chain(activations[activations.length-(l+1)])
													.transpose().done()
											).done();
		  
		}

		return [nabla_b, nabla_w];
	};
	
	var update_mini_batch = function(mini_batch, eta, lambda, n) {

		var nabla_b = zeroMatrixCopyOf(self.biases);
		var nabla_w = zeroMatrixCopyOf(self.weights);

		for (var m = 0; m < mini_batch.length; m++) {
		  var x = mini_batch[m].value, 
		      y = mini_batch[m].target; 

		  var backprop_result = backprop(x, y);
		  var delta_nabla_b = backprop_result[0],
		      delta_nabla_w = backprop_result[1];

		  nabla_b = nabla_b.map(function(x, i) { return math.add(nabla_b[i], delta_nabla_b[i]) });
		  nabla_w = nabla_w.map(function(x, i) { return math.add(nabla_w[i], delta_nabla_w[i]) });
		  
		  self.biases = d3.zip(self.biases, nabla_b)
		                  .map(function(x, i) {
		                    var b = x[0], 
		                    	nb = x[1],
		                    	bias_delta = math.multiply(nb, parseFloat(eta)/mini_batch.length);
		                    return math.subtract(b, bias_delta);
		                  });
		  self.weights = d3.zip(self.weights, nabla_w)
		                   .map(function(x, i) {
		                     var w = x[0], 
		                     	 nw = x[1],
			                     	//encourages smaller weights in network (less sensitive to noise)
			                     regularizedWeights = math.multiply(1-eta*(lambda/n), w), 
			                     weightDelta = math.multiply(nw, parseFloat(eta)/mini_batch.length);

			                 return math.subtract(regularizedWeights, weightDelta);
		                   });
		}
	};

	var cost_derivative = function(output_activations, targets) {
		return linearError(output_activations, targets);
	};

	var linearError = function(output_activations, targets) {
		return  math.chain(output_activations)
					.subtract(targets)
					.done();
	};

	var logisticError = function(output_activations, targets) {
		return math.dotMultiply.call(
	                  math.subtract(output_activations, targets),
	                  output_activations,
	                  math.subtract(1, output_activations)
	                );
	}

	var softmaxError = function(output_activations, targets) {
	  return math.dotMultiply(
	                  math.subtract(output_activations, targets),
	                  math.subtract(
	                        output_activations,
	                        math.multiply(math.square(output_activations), -1)
	                  )
	                );
	}



	var zeroMatrixCopyOf = function(matrix) {
		return matrix.map(function(x) { 
		                    	return math.matrix(math.zeros(x.size()))
		                  	});
	};
	var oneMatrixCopyOf = function(matrix) {
		return matrix.map(function(x) { 
								return math.matrix(math.ones(x.size())) 
							});
	};

	var sigmoid = function(z) { return 1.0/(1.0 + Math.exp(-z)) };
	var sigmoid_prime = function(z) { return sigmoid(z)*(1-sigmoid(z)) };

	var sigmoid_vec = function(v) {return v.map(sigmoid) };
	var sigmoid_prime_vec = function(v) {return v.map(sigmoid_prime) };

	

	self.init(sizes);

	return self;

};