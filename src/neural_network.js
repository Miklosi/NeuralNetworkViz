//var Network 

var NeuralNetwork = function(sizes) { 

	var self = this;

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

		/*
		//sizes: [4,5,6,7] //no. neurons at each layer
		self.biases = self.sizes.slice(1, self.sizes.length)
		                   .map(function(x, i) { 
		                   	  return matrixApply(math.ones([x, 1]), initialiseFn);
		                    });
		self.weights = self.sizes.slice(1, self.sizes.length)
		                   .map(function(x, i) { 
								return matrixApply(math.ones([x, sizes[i]]), initialiseFn)
		                    });
		*/

		//TEST DATA - replace math.matrix with array keyword in IPython
		//https://www.pythonanywhere.com/try-ipython/
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

	};

	//https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py
	//training_data: [{ value: x, target: y },...]
	//eta: learning rate (0.25)
	NeuralNetwork.prototype.SGD = function(training_data, epochs, mini_batch_size, eta) {//,test_data=None
		var n = training_data.length;

		for (var j = 1; j <= epochs; j++) {
			var epoch_training_data = d3.shuffle(training_data);

		  	var mini_batches = [];
		  	for (var k=0; k < n; k += mini_batch_size) {
		    	mini_batches.push(epoch_training_data.slice(k, k+mini_batch_size));
		  	}

		  	for (var m = 0; m < mini_batches.length; m++) {
		    	update_mini_batch(mini_batches[m], eta);
		  	}

		  	console.log("Epoch " + j + " complete");
		}
	};

	//*
	NeuralNetwork.prototype.evaluate = function(test_data) {
		//return (for test_data) the no. cases where self.feed_forward(inputs) == test_data.outputs

		var num_correct = test_data.map(function(d, i) {

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
			return feedForward(d.value)._data[0][0] == d.target[0][0] ? 1 : 0;
 		});

		return parseFloat(d3.sum(num_correct)) / test_data.length;
	};

	var feedForward = function(a) { //a: inputs, eg.: [[1],[1]]
		for (var i = 0; i < self.weights.length; i++) {
		  a = sigmoid_vec(
		        math.add(
		          math.multiply(self.weights[i], a), 
		          self.biases[i]
		        )
		      ); 
		}
		return a;
	};

	var backprop = function(x, y) {

		var nabla_b = zeroMatrixCopyOf(self.biases);
		var nabla_w = zeroMatrixCopyOf(self.weights);

		//feed-foward
		var activation = x,
		    activations = [x],
		    zs = [];

		for (var k = 0; k < self.biases.length; k++) {
		    var w = self.weights[k], 
		        b = self.biases[k];
		    var z = math.add(math.multiply(w, activation), b);

		    zs.push(z);
		    activations.push(sigmoid_vec(z));
		}

		//backward-pass
		    //delta = cost_derivative(activations[-1], y) * sigmoid_prime_vec(zs[-1])
		var delta = math.dotMultiply(
								  cost_derivative(activations[activations.length-1], y),
		              			  sigmoid_prime_vec(zs[zs.length-1])
		              			 );
		nabla_b[nabla_b.length-1] = delta;
		nabla_w[nabla_w.length-1] = math.multiply(delta, math.transpose(activations[activations.length-2]));

		for (var l = 2; l < self.num_layers; l++) {
			var z = zs[zs.length-l];
			var spv = sigmoid_prime_vec(z);

			//delta = np.dot(self.weights[-l+1].transpose(), delta) * spv
			delta = math.dotMultiply(math.multiply(math.transpose(self.weights[self.weights.length-(l-1)]), delta), spv);

			//nabla_b[-l] = delta
			nabla_b[nabla_b.length-l] = delta; 

			////nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
			nabla_w[nabla_w.length-l] = math.multiply(delta, math.transpose(activations[activations.length-l]));
		  
		}

		return [nabla_b, nabla_w];
	};
	
	var update_mini_batch = function(mini_batch, eta) {
		//console.log(math.format(mini_batch));

		var nabla_b = zeroMatrixCopyOf(self.biases);
		var nabla_w = zeroMatrixCopyOf(self.weights);

		for (var m = 0; m < mini_batch.length; m++) {
		  var x = mini_batch[m].value, 
		      y = mini_batch[m].target; 

		  var backprop_result = self.backprop(x, y);
		  var delta_nabla_b = backprop_result[0],
		      delta_nabla_w = backprop_result[1];

		  //temp solution until self.backprop() is written!!
		  //var delta_nabla_b = oneMatrixCopyOf(self.biases);
		  //var delta_nabla_w = oneMatrixCopyOf(self.weights);

		  nabla_b = nabla_b.map(function(x, i) { return math.add(nabla_b[i], delta_nabla_b[i]) });
		  nabla_w = nabla_w.map(function(x, i) { return math.add(nabla_w[i], delta_nabla_w[i]) });
		  
		  self.biases = d3.zip(self.biases, nabla_b)
		                  .map(function(x, i) {
		                    var b = x[0], nb = x[1];
		                    return math.subtract(b, math.multiply(nb, parseFloat(eta)/mini_batch.length));
		                  });

		  self.weights = d3.zip(self.weights, nabla_w)
		                   .map(function(x, i) {
		                     var w = x[0], nw = x[1];
		                     return math.subtract(w, math.multiply(nw, parseFloat(eta)/mini_batch.length));
		                   });
		}
	};

	var cost_derivative = function(output_activations, y) {
		//todo: add different cost_derivative formulas (see bottom of file)
		return math.subtract(output_activations, y);
	};

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

/*
function linearError(activations) {
  return math.dotMultiply(
                math.subtract(activations, targets),
                1/nData);
}

function logisticError(activations) {
  return math.dotMultiply.call(
                  math.subtract(activations, targets),
                  activations,
                  math.subtract(1, activations)
                );
}

function softmaxError(activations) {
  return math.dotMultiply.call(
                  math.subtract(activations, targets),
                  math.subtract(
                        activations,
                        math.dotMultiply.call(activations, activations, -1)
                  ),
                  1/nData
                );
}
*/