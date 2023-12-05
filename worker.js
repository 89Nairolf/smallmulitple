importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs");
var model;


async function initNetwork(model, firstLayer, secondLayer, sameWeights, classifier) {
    model = tf.sequential();

    function setinitWeights(a,b,layer){
      let weight_array = []
      for (let index_a = 0; index_a < a; index_a++){
        for (let index_b = 0; index_b < b; index_b++){
          weight_array.push((1+index_b)/b)
        }
      }
      let tensor = tf.tensor(weight_array,[a,b])
      model.layers[layer].setWeights([tensor,tf.zeros([b])])
    }

    if (classifier){
    model.add(tf.layers.dense({ inputShape: [2], useBias: true, units: firstLayer, activation: "sigmoid" }));
    model.add(tf.layers.dense({ units: secondLayer, useBias: true, activation: "sigmoid" }));
    model.add(tf.layers.dense({ units: 2, useBias: true, activation: "softmax" }));
    }else{
    model.add(tf.layers.dense({ inputShape: [2], useBias: true, units: firstLayer, activation: "tanh" }));
    model.add(tf.layers.dense({ units: secondLayer, useBias: true, activation: "tanh" }));
    model.add(tf.layers.dense({ units: 1, useBias: true, activation: "tanh" }));
    }
    if (sameWeights == 0){
      let init_weights_input = tf.fill([2, firstLayer], 0.5)
      let init_weights_first = tf.fill([firstLayer, secondLayer], 0.5)
      let init_weights_second = tf.fill([secondLayer, 2], 0.5)
      model.layers[0].setWeights([init_weights_input,tf.zeros([firstLayer])])
      model.layers[1].setWeights([init_weights_first,tf.zeros([secondLayer])])
      model.layers[2].setWeights([init_weights_second,tf.zeros([2])])
    }
    else if (sameWeights == 1){
      setinitWeights(2,firstLayer,0);
      setinitWeights(firstLayer,secondLayer,1);
      setinitWeights(secondLayer,2,2);
    }   
    
    model.compile({ loss: "meanSquaredError", optimizer: tf.train.adam(.01), metrics: ["acc"] });
    return model
  }


async function training(model, id, trainingData, outputData,classifier) {
  let acc = [];
  let learningRate = [];
  let prevLayer = [await model.getWeights()[0].array(), await model.getWeights()[2].array(), await model.getWeights()[4].array()]
  const startTime = Date.now();
  const h = await model.fit(trainingData, outputData, {
    epochs: 50,
    validationSplit: 0.2,
    callbacks: [{
      onEpochEnd: async (epoch, logs) => {
        let input = await model.getWeights()[0].array()
        let firstLayer = await model.getWeights()[2].array()
        let secondLayer = await model.getWeights()[4].array()
        if (classifier){
          acc.push({ epoch: epoch, value: logs.val_loss, type: "val_loss" });
        acc.push({ epoch: epoch, value: logs.loss, type: "loss" });
        acc.push({ epoch: epoch, value: logs.val_acc, type: "val_acc" });
        acc.push({ epoch: epoch, value: logs.acc, type: "acc" });
      }else{
      acc.push({ epoch: epoch, value: logs.val_loss < 0.5? logs.val_loss: 0.5, type: "val_loss" });
      acc.push({ epoch: epoch, value: logs.loss < 0.5? logs.loss: 0.5, type: "loss" });
    }

        learningRate_values = await learningRateDecay( prevLayer, input, firstLayer, secondLayer)
        learningRate.push({epoch:epoch, value:learningRate_values[0], type:"input"})
        learningRate.push({epoch:epoch, value:learningRate_values[1], type:"hidden1"})
        learningRate.push({epoch:epoch, value:learningRate_values[2], type:"hidden2"})
        prevLayer = [input, firstLayer, secondLayer]

        if (epoch%2==0){
        self.postMessage([id, "acc", acc])
        self.postMessage([id, "lrd" , learningRate])}
        if (epoch%10 == 0){
          showheatmap(id, input, firstLayer, secondLayer, classifier)
        }
        }

    },
      //callback
    ]
  });
  let switched_signs = await similarity([await model.getWeights()[0].array(),await model.getWeights()[2].array(),await model.getWeights()[4].array()])
  self.postMessage([id, "similarity", switched_signs])
  let time = Date.now() - startTime
}

async function showheatmap(id, input, firstLayer, secondLayer, classifier){
  let data_weights = []; 
  let all_weights = [input, firstLayer, secondLayer]
  let layer_name =""
  let max_node_weight  = 0 
  for (let layer = 0; layer < 3; layer++){
    let number_neurons = all_weights[layer].length;
    for (let neuron = 0; neuron < number_neurons; neuron++){
      let weights = all_weights[layer][neuron].length;
      let weight_sum = 0;
      for (let weight = 0; weight < weights; weight++){
        weight_sum += Math.abs(all_weights[layer][neuron][weight]);
      } 
      if (weight_sum > max_node_weight){max_node_weight = weight_sum}
    } 
  }
  for (let layer = 0; layer < 3; layer++){
    if (layer == 0){ layer_name = "Input Layer"}
    else if (layer == 1){layer_name ="First Hidden Layer"}
    else if (layer == 2){layer_name ="Second Hidden Layer"}
    let number_neurons = all_weights[layer].length;
    for (let neuron = 0; neuron < number_neurons; neuron++){
      let weights = all_weights[layer][neuron].length;
      let weight_sum = 0;
      for (let weight = 0; weight < weights; weight++){
        weight_sum += Math.abs(all_weights[layer][neuron][weight]);
      }
      let scaled_weight_sum = weight_sum/max_node_weight
      let yposition = ((neuron+1)/(number_neurons+1));
      data_weights.push({"layer":layer_name,"yposition":yposition, "weight":scaled_weight_sum,"neuron":neuron})
    }

  }
  if (classifier){
  data_weights.push({"layer":"Output Layer","yposition":0.33, "weight":0.7,"neuron":0});
  data_weights.push({"layer":"Output Layer","yposition":0.67, "weight":0.7,"neuron":1});}
  else{
    data_weights.push({"layer":"Output Layer","yposition":0.5, "weight":0.7,"neuron":0});
  }
  self.postMessage([id, "heatmap", data_weights])
}


async function learningRateDecay( prevLayer, input, firstLayer, secondLayer){
  function diff(prevLayer, actLayer){
    let lRateDecay = 0;
    for (let i = 0; i< actLayer.length; i++){
      for (let j = 0;j<actLayer[0].length;j++){
      lRateDecay += Math.abs(prevLayer[i][j]-actLayer[i][j]);
    }}
    return (lRateDecay/actLayer.length)
  }
  let layer_input = diff(prevLayer[0], input);
  let layer_H1 =diff(prevLayer[1], firstLayer);
  let layer_H2 =diff(prevLayer[2], secondLayer);
  
  return [layer_input < 0.5?layer_input:0.5,layer_H1< 0.5?layer_H1:0.5,layer_H2< 0.5?layer_H2:0.5]

}

async function similarity(layers){
function sort_dissimilarity_revers(d2array) {
  let minimum = d2array[0].length * 2;
  let clothesthneighbour = 1;
  let d2arraylen = d2array.length;
  let d2outarraylen = d2array[0].length;
  let reverse = false;
  for (let outarray = 1; outarray < d2arraylen; outarray++) {
    let euklidsum = 0;
    let euklidsum_revers = 0;
    for (let inarray = 0; inarray < d2outarraylen; inarray++) {
      euklidsum += Math.abs(d2array[0][inarray] - d2array[outarray][inarray]);
      euklidsum_revers += Math.abs(d2array[0][inarray] + d2array[outarray][inarray]);
    }
    if (euklidsum < euklidsum_revers) {
      if (euklidsum < minimum) {
        minimum = euklidsum;
        clothesthneighbour = outarray;
        reverse = false;
      }
    }
    else if (euklidsum_revers < minimum) {
      minimum = euklidsum_revers;
      clothesthneighbour = outarray;
      reverse = true;
    }
  }
  let scaled_minimum = Math.floor(10*(1-(minimum/d2outarraylen)))/10
  return [clothesthneighbour, reverse, scaled_minimum.toString()]
}

function scaleData(predata){
//scale down to 1max
//create 1D array with max/min
let maxRow = predata.map(function (row) { return Math.max.apply(Math, row); });
let minRow = predata.map(function (row) { return Math.min.apply(Math, row); });
let max_row_length = maxRow.length;
var scaleddata = [];
//scale with Math.max(maxRow, Math.abs(minRow))
for (let i = 0; i < max_row_length; i++) {
  if (Math.abs(maxRow[i]) > Math.abs(minRow[i])) {
    var scale = Math.abs(maxRow[i]);
  }
  else {
    var scale = Math.abs(minRow[i]);
  }
  var helper = predata[i].map(x => x / scale);
  scaleddata.push(helper);
}
return scaleddata
}

let similarity_array = []
for (let i = 0; i < layers.length; i++){
let switched_signs = scaleData(layers[i]);
let iterations = switched_signs.length - 1;
for (let index = 0; index < iterations; index++) {
  var result = sort_dissimilarity_revers(switched_signs.slice(index, switched_signs.length));
  if (result[0] != 1) {
    let rowhelper3 = switched_signs[result[0] + index];
    switched_signs[result[0] + index] = switched_signs[index + 1];
    switched_signs[index + 1] = rowhelper3;
  }
  if (result[1]) {
    switched_signs[index + 1] = await switched_signs[index + 1].map(el => -el);
  }
  similarity_array.push({"layerID":i, "similarity":result[2]});
}
}
return similarity_array
}




  self.onmessage = async (e) => {
    const { id, firstLayer, secondLayer, sameWeights, data, classifier} = e.data;
    const trainingData = tf.tensor2d(data.map(item => [item.x, item.y]));
    //e{} -> {} after message ? 
    const outputData = classifier ? tf.tensor(data.map(item => [
      item.label == 1 ? 1 : 0,
      item.label == 2 ? 1 : 0,
    ])):tf.tensor(data.map(item => [item.label]));
    model = await initNetwork(model, firstLayer, secondLayer, sameWeights, classifier);
    await training(model, id, trainingData, outputData, classifier)
  }