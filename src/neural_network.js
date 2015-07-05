//Code adapted from 
//https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py
	
var NeuralNetwork = function(options) { 

	var self = this;

	this.options = options || {};

	this.learningRate = options.learningRate || 0.3;
  	this.momentum = options.momentum || 0.1;
  	this.hiddenSizes = options.hiddenLayers;
  	this.binaryThresh = options.binaryThresh || 0.5;
	this.sizes = options.sizes || [2,2,1];
	this.num_layers = options.sizes.length;

	this.initialize(this.options.sizes);


	this.linear = function(z) { return z }
	this.linear_prime = function(z) { return 1.0 }
	this.sigmoid = function(z) { return 1.0/(1.0 + Math.exp(-z)) };
	this.sigmoid_prime = function(z) { return self.sigmoid(z)*(1-self.sigmoid(z)) };
	this.gaussian = function(z) { return Math.exp(-Math.pow(z,2)) };
	this.gaussian_prime = function(z) { return -2*z*this.gaussian(z) };
	this.rational_sigmoid = function(z) { return z / (1.0 + z * z) };
	this.rational_sigmoid_prime = function(z) { return Math.sqrt(1.0 + z * z) };

	this.softmax_vec = function(v) { 
		var v_exp = v.map(function(z) { return Math.exp(z)}),
			v_sum = d3.sum(v_exp);
		return v_exp.map(function(z) { return z/v_sum; });
	 }

	this.linear_vec = function(v) {return v.map(self.linear) };	
	this.linear_prime_vec = function(v) {return v.map(self.linear_prime) };	
	this.sigmoid_vec = function(v) {return v.map(self.sigmoid) };
	this.sigmoid_prime_vec = function(v) {return v.map(self.sigmoid_prime) };
	this.gaussian_vec = function(v) {return v.map(self.gaussian) };
	this.gaussian_prime_vec = function(v) {return v.map(self.gaussian_prime) };
	this.rational_sigmoid_vec = function(v) {return v.map(self.rational_sigmoid) };
	this.rational_sigmoid_prime_vec = function(v) {return v.map(self.rational_sigmoid_prime) };

};


NeuralNetwork.prototype = {

	//sizes: no. neurons at each layer
	//e.g. [2,2,1] = [2 (input), 2 (hidden layer1), 1 (output)]
	//for truth-tables: [2,2,1]
	//for character-recognition: [784, 30, 10]
	initialize: function() {

		var sizes = this.options.sizes;
		
		var initialiseFn =  function() { 
								return 2*(Math.random()-.5) /*(Math.random() -.5) * 2/Math.sqrt(sizes[0])*/; 
							};

		var matrixApply = function(m, FUN) {
			return m.map(function(v) { 
							return v.map(function(w) { return FUN() }) 
						});
		};

		this.biases = this.sizes.slice(1, this.sizes.length)
		                    .map(function(x, i) { 
		                   	  return math.matrix(matrixApply(math.ones([x, 1]), initialiseFn));
		                    });
		this.weights = this.sizes.slice(1, this.sizes.length)
		                   .map(function(x, i) { 
								return math.matrix(matrixApply(math.ones([x, sizes[i]]), initialiseFn));
		                    });
		return this;
	},

	//training_data: [{ value: x, target: y },...]
	//eta: learning rate (0.25)
	train: function(training_data, epochs, mini_batch_size, eta, test_data) {//,test_data=None
		var n = training_data.length;
		var lambda = lambda || 0;

		for (var j = 1; j <= epochs; j++) {
			var epoch_training_data = training_data; //d3.shuffle(training_data);

		  	var mini_batches = [];
		  	for (var k=0; k < n; k += mini_batch_size) {
		    	mini_batches.push(epoch_training_data.slice(k, k+mini_batch_size));
		  	}

		  	for (var m = 0; m < mini_batches.length; m++) {
		    	this.update_mini_batch(mini_batches[m], eta, lambda, training_data.length);
		  	}

		  	if(test_data)
		  		console.log("Epoch %d: %f / %f", j, this.test(test_data), n);
		  	else
		  		console.log("Epoch %d complete", j);
		}

		return this;
	},

	//return (for test_data) the no. cases where this.feed_forward(inputs) == test_data.outputs
	test: function(test_data) {
		var self = this;

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
				actual = self.feedForward(d.value)._data[0][0];

			//console.log("input: %s, output: %f, target: %f", math.format(d.value), actual, target);

			return Math.round(actual.toFixed(3)) == Math.round(target.toFixed(3)) ? 1 : 0;
 		}));
	},

	feedForward: function(a) { //a: inputs, eg.: [[1],[1]]
		for (var i = 0; i < this.weights.length; i++) {
			var z = math.chain(this.weights[i])
					.multiply(a)
					.add(this.biases[i]).done();
//			if (i == (this.weights.length - 1))
//				a = this.softmax_vec(z);
//			else
		  		a = this.sigmoid_vec(z); 
		}
		return a;
	},

	backprop: function(x, y) {

		var nabla_b = math.zeroMatrixCopyOf(this.biases);
		var nabla_w = math.zeroMatrixCopyOf(this.weights);

		//feed-foward: get activations & component z-scores (weighted sum of inputs)
		var activation = x,
		    activations = [x],
		    zs = [];

		for (var k = 0; k < this.biases.length; k++) {
		    var w = this.weights[k], 
		        b = this.biases[k];
		    var z = math.chain(w)
		    			.multiply(activation)
		    			.add(b).done();

		    zs.push(z);

//			if (k == (this.weights.length - 1))
//				activations.push(this.softmax_vec(z));
//			else
		  		activations.push(this.sigmoid_vec(z));

		    //activations.push(this.sigmoid_vec(z));
		}

		//output error: (via cost derivative), how inaccurate are activations vs. targets?
		var delta = math.chain(this.cost_derivative(activations[activations.length-1], y))
						.dotMultiply(this.sigmoid_prime_vec(zs[zs.length-1])).done();

		nabla_b[nabla_b.length-1] = delta;
		nabla_w[nabla_w.length-1] = math.chain(delta)
										.multiply(//multiply
											math.transpose(activations[activations.length-2])
										).done();

		//backpropagate error: determine output error contributions from each previous layer
		for (var l = 2; l < this.num_layers; l++) {
			var z = zs[zs.length-l],
				spv = this.sigmoid_prime_vec(z);

			//error at layer 'l'
			delta = math.chain(this.weights[this.weights.length-(l-1)])
						.transpose()
						.multiply(delta)
						.dotMultiply(spv).done();

			nabla_b[nabla_b.length-l] = delta; 
			nabla_w[nabla_w.length-l] = math.chain(delta)
											.multiply(//multiply
												math.transpose(activations[activations.length-(l+1)])
											).done();
		  
		}

		return [nabla_b, nabla_w];
	},





/*
runInput: function(input) {
    this.outputs[0] = input;  // set output state of input layer

    for (var layer = 1; layer <= this.outputLayer; layer++) {
      for (var node = 0; node < this.sizes[layer]; node++) {
        var weights = this.weights[layer][node];

        var sum = this.biases[layer][node];
        for (var k = 0; k < weights.length; k++) {
          sum += weights[k] * input[k];
        }
        this.outputs[layer][node] = 1 / (1 + Math.exp(-sum));
      }
      var output = input = this.outputs[layer];
    }
    return output;
  },
*/





	update_mini_batch: function(mini_batch, eta, lambda, n) {

		var nabla_b = math.zeroMatrixCopyOf(this.biases);
		var nabla_w = math.zeroMatrixCopyOf(this.weights);

		for (var m = 0; m < mini_batch.length; m++) {
		  var x = mini_batch[m].value, 
		      y = mini_batch[m].target; 

		  var backprop_result = this.backprop(x, y);
		  var delta_nabla_b = backprop_result[0],
		      delta_nabla_w = backprop_result[1];

		  nabla_b = nabla_b.map(function(x, i) { return math.add(nabla_b[i], delta_nabla_b[i]) });
		  nabla_w = nabla_w.map(function(x, i) { return math.add(nabla_w[i], delta_nabla_w[i]) });
		  
		  this.biases = d3.zip(this.biases, nabla_b)
		                  .map(function(x, i) {
		                    var b = x[0], 
		                    	nb = x[1],
		                    	bias_delta = math.multiply(nb, parseFloat(eta)/mini_batch.length);
		                    return math.subtract(b, bias_delta);
		                  });
		  this.weights = d3.zip(this.weights, nabla_w)
		                   .map(function(x, i) {
		                     var w = x[0], 
		                     	 nw = x[1],
			                     	//encourages smaller weights in network (less sensitive to noise)
			                     regularizedWeights = math.multiply(1-eta*(lambda/n), w), 
			                     weightDelta = math.multiply(nw, parseFloat(eta)/mini_batch.length);

			                 return math.subtract(w, weightDelta);
		                   });
		}
	},

	cost_derivative: function(output_activations, targets) {
		return math.chain(output_activations)
					.subtract(targets)
					.done();
	}
};


math.zeroMatrixCopyOf = function(matrix) {
	return matrix.map(function(x) { 
    	return math.matrix(math.zeros(x.size()))
  	});
};
math.oneMatrixCopyOf = function(matrix) {
	return matrix.map(function(x) { 
		return math.matrix(math.ones(x.size())) 
	});
};

