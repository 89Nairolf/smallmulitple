
importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs");
var model;

let epochs = 51

let weights_layer1 = [];
let weights_layer1_array = [];
let data_weights_array = [[],[],[]];

async function initNetwork(model, firstLayer, secondLayer, sameWeights, classifier) {
    model = tf.sequential();
    weights_layer1_array = [];

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
    for (let i =0; i < firstLayer;i++){
      weights_layer1_array.push([])
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
      model.layers[0].setWeights([init_weights_input,tf.zeros([firstLayer])])
      model.layers[1].setWeights([init_weights_first,tf.zeros([secondLayer])])
      if (classifier){
      let init_weights_second = tf.fill([secondLayer, 2], 0.5)
      model.layers[2].setWeights([init_weights_second,tf.zeros([2])])
      }
      else{
        let init_weights_second = tf.fill([secondLayer, 1], 0.5)
        model.layers[2].setWeights([init_weights_second,tf.zeros([1])])}
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
  let loss = [];
  let acc_evo = []
  let lrdRate = [[],[],[]]
  let learningRate = [];
  weights_layer1 = [];
  let prevLayer = [await model.getWeights()[0].array(), await model.getWeights()[2].array(), await model.getWeights()[4].array()]
  const h = await model.fit(trainingData, outputData, {
    epochs: epochs,
    validationSplit: 0.2,
    callbacks: [{
      onEpochEnd: async (epoch, logs) => {
        
        let input = await model.getWeights()[0].array()
        let firstLayer = await model.getWeights()[2].array()
        let secondLayer = await model.getWeights()[4].array()
        loss.push(logs.loss)
        
        if (classifier){
          acc.push({ epoch: epoch, value: logs.val_loss, type: "val_loss" });
        acc.push({ epoch: epoch, value: logs.loss, type: "loss" });
        acc.push({ epoch: epoch, value: logs.val_acc, type: "val_acc" });
        acc.push({ epoch: epoch, value: logs.acc, type: "acc" });
        acc_evo.push(logs.acc)
      }else{
      acc.push({ epoch: epoch, value: logs.val_loss < 7? logs.val_loss: 0.7, type: "val_loss" });
      acc.push({ epoch: epoch, value: logs.loss < 0.7? logs.loss: 0.7, type: "loss" });
    }
        pushWeightsLayer1(firstLayer,epoch);
        learningRate_values = await learningRateDecay( prevLayer, input, firstLayer, secondLayer)
        learningRate.push({epoch:epoch, value:learningRate_values[0], type:"input"})
        learningRate.push({epoch:epoch, value:learningRate_values[1], type:"hidden1"})
        learningRate.push({epoch:epoch, value:learningRate_values[2], type:"hidden2"})
        lrdRate[0].push(learningRate_values[0])
        lrdRate[1].push(learningRate_values[1])
        lrdRate[2].push(learningRate_values[2])
        prevLayer = [input, firstLayer, secondLayer]

        if (epoch%2==0){
          self.postMessage([id, "acc", acc])
          self.postMessage([id, "lrd" , learningRate])
          self.postMessage([id, "firstLayer", weights_layer1])
        }
        if (epoch%10 == 0){
          let data_weights = await showheatmap(id, input, firstLayer, secondLayer, classifier)
          self.postMessage([id, "heatmap", data_weights])

          //let timer_similarity_start = performance.now()
          //let similarity_array = await similarity(id, [input, firstLayer, secondLayer])
          //timer_similarity_end += performance.now() - timer_similarity_start
          //self.postMessage([id, "similarity", similarity_array])
        }
        }

    },
      //callback
    ]
  });
  //let switched_signs = await similarity(id, [await model.getWeights()[0].array(),await model.getWeights()[2].array(),await model.getWeights()[4].array()])
  let tooltip_array = [[],[],[],[]]

  //chart 1 loss
  let loss1 = []
  let loss_der2 = [];
  let reduce = "";
  for (let i = 1; i < loss.length; i++){
    loss1.push(loss[i-1]-loss[i])
  }
  for (let i = 1; i < loss1.length; i++){
        loss_der2.push(Math.round(1000*(loss1[i-1]-loss1[i])))

  }
  let curversaver = [{"epoch":0, "value":0}]
  for (let i = 5; i < loss_der2.length-5; i++){
    let curve = 0;
    let sumpositive = 0;
    let sumnegative = 0;
    
    for (let j=0; j < 5; j++){  
      if (loss_der2[i-j] <= 0){
        curve++
        sumnegative += loss_der2[i-j]
      }
      if (loss_der2[i+j] >= 0){
        curve++
        sumpositive += loss_der2[i+j]
      }     
    }
    if (curve >= 8 && sumnegative < -3 && sumpositive > 3 && loss_der2[i-1]<=0 &&loss_der2[i]>=0){
      if (sumnegative+sumpositive > curversaver.value){
        curversaver.epoch = i+1
        curversaver.value = Math.abs(sumnegative)+sumpositive
      }
      reduce = "Reduces training at epoch "+ (i+1)+".<br>"
      }
  }
  if (reduce == ""){
    tooltip_array[0] += ("The network was not able to learn<br> or can still improve.<br>")}else{
  }
  tooltip_array[0] += (reduce)
  //add together with all information
  if (curversaver.epoch != 0){
    //self.postMessage(id, "")
  }

  if (classifier){
  //explain: acc, local min
  let streak = 0
  acc_evo.push(0)
  for (let i = 3; i < acc_evo.length-2; i++){

    let localmin = true;
    for (let j=0; j < 6; j++){
    if (0.02 < Math.abs(acc_evo[i-3+j] - acc_evo[i])){
      localmin = false
    }
  }
    if (streak > 0){
      if (localmin){
        streak++
      }else{
        if (acc_evo[i+1] > .998){ 
          let globalmin = "A global minima was found at epoch: "+ (i-3-streak) +".<br>"
          tooltip_array[0]+=(globalmin)
        } else{
          let localm = "A local minima was found in " +((i-3-streak+ i+1)/2) + ".<br>"
          tooltip_array[0]+=(localm)
        i+=2}
        streak=0

      }
    }else{      
      if(localmin){
        streak=1
      }else{
        streak=0
      }
    }}

  }
//console.log("chart 2:what neuron highest influence; many neurons in following layer?, average layer weight")
let average = [[],[],[]]
let position = [[],[]]
for (let i in data_weights_array){
  let counter = 0
    let sum = 0
  for (let j in data_weights_array[i]){
    sum += data_weights_array[i][j]
    if (data_weights_array[i][j] == 1){
      counter++
      position[0].push(i)
      position[1].push(j)
  }
}
    average[i].push(sum/data_weights_array[i].length)
    if (counter == data_weights_array[i].length && counter > 1){
      tooltip_array[1]+=("All neurons in the layer " +i+" learn the same.<br>")
    }
}
if (position[0].length == 1){
  let maxposition  = ("Max neuron is in layer "+ position[0][0] + " and position " + position[1][0]+".<br>")
  console.log(maxposition)
  tooltip_array[1]+=(maxposition)
}
//max 0
if (average[0] > average[1] && average[1] > average[2]){tooltip_array[1]+=("The neurons in the input layer have the<br> highest average influence, followed<br> by the first layer.<br>")}
if (average[0] > average[1] && average[1] < average[2] && average[0] > average[2]){tooltip_array[1]+=("The neurons in the input layer have the<br> highest average influence, followed<br> by the second layer.<br>")}
//max 2
if (average[0] < average[1] && average[1] < average[2]){tooltip_array[1]+=("The neurons in the second layer have<br> the highest average influence, followed<br> by the first layer.<br>")}
if (average[0] > average[1] && average[1] < average[2]&& average[0] < average[2]){tooltip_array[1]+=("The neurons in the second layer have<br> the highest average influence, followed<br> by the input layer.<br>")}
//max 1
if (average[0] < average[1] && average[1] > average[2]&& average[0] < average[2]){tooltip_array[1]+=("The neurons in the first layer have<br> the highest average influence, followed<br> by the second layer.<br>")}
if (average[0] < average[1] && average[1] > average[2]&& average[0] > average[2]){tooltip_array[1]+=("The neurons in the first layer have<br> the highest average influence, followed<br> by the input layer.<br>")}

//learning curve layer 1
let first_der = []
let second_der = []
let second_der_max = []
let highest_first = []
for (let i=0; i < weights_layer1_array.length; i++){
  first_der.push([])
  for (let j=1; j < weights_layer1_array[i].length; j++){
    first_der[i].push(weights_layer1_array[i][j]-weights_layer1_array[i][j-1])
  }
}
for (let i=0; i < first_der.length; i++){
  highest_first.push(Math.max(...first_der[i]))
  second_der.push(0)
  second_der_max.push(0)
  for (let j=1; j < first_der[i].length; j++){
    let deriv = Math.abs(first_der[i][j]-first_der[i][j-1])
    second_der[i] +=deriv
    if (second_der_max[i] < deriv){
      second_der_max[i] = deriv
    }
  }
}
let highestslope = highest_first.indexOf(Math.max(...highest_first))
let sum_second = second_der.indexOf(Math.max(...second_der))
let max_second = second_der_max.indexOf(Math.max(...second_der_max))
let textanalyse = ("The highest slope has Neuron "+ highestslope + ";<br>the heighest overal change of adjusting to<br> a minimal error configurations has Neuron " + sum_second+ ";<br>the heighest overal change of adjusting to<br> a minimal error configurations has Neuron "+ max_second)
tooltip_array[2]+=(textanalyse)

console.log("total learning rate, what neuron learned")
//console.log("a high overall change -> unglatten chart 4")


/* smoothen graph
var array = [10, 13, 7, 11, 12, 9, 6, 5];

function smooth(values, alpha) {
  var weighted = average(values) * alpha;
  var smoothed = [];
  for (var i in values) {
      var curr = values[i];
      var prev = smoothed[i - 1] || values[values.length - 1];
      var next = curr || values[0];
      var improved = Number(this.average([weighted, prev, curr, next]).toFixed(2));
      smoothed.push(improved);
  }
  return smoothed;
}

function average(data) {
  var sum = data.reduce(function(sum, value) {
      return sum + value;
  }, 0);
  var avg = sum / data.length;
  return avg;
}

smooth(array, 0.85); 
*/

//console.log("chart 4: max layer 1,2,3")
let maxvalues = [[],[],[]]
let positions = []
for (let i = 0; i < lrdRate.length; i++){
  for (let j = 4; j < lrdRate[i].length-4; j++){
    if (
      lrdRate[i][j] > lrdRate[i][j - 1] &&
      lrdRate[i][j] > lrdRate[i][j - 2] &&
      lrdRate[i][j] > lrdRate[i][j - 3] &&
      lrdRate[i][j] > lrdRate[i][j - 4] &&
      lrdRate[i][j] > lrdRate[i][j + 1] &&
      lrdRate[i][j] > lrdRate[i][j + 2] &&
      lrdRate[i][j] > lrdRate[i][j + 3] &&
      lrdRate[i][j] > lrdRate[i][j + 4]
    ){maxvalues[i].push(lrdRate[i][j]); j+=4}
    else if (j == lrdRate[i].length-5){
      maxvalues[i].push( Math.max(lrdRate[i][j],lrdRate[i][j+1],lrdRate[i][j+2],lrdRate[i][j+3],lrdRate[i][j+4]))
    }
  }
  if (maxvalues[i].length > 1){
    let maximum = Math.max(...maxvalues[i])*0.75
    maxvalues[i] = maxvalues[i].filter((element) => element > maximum)
  }
  
  for (let j = 0; j < maxvalues[i].length; j++){
    let text = "A maximum in layer "+ i+ "<br> was found at epoch "+lrdRate[i].indexOf(maxvalues[i][j])+".<br>"
    tooltip_array[3]+=(text)
    positions.push(lrdRate[i].indexOf(maxvalues[i][j]))
  }
}
let sortposition = positions.slice().sort((a, b) => a - b);
for (let j = 0; j < sortposition.length-2;j++){
  if (sortposition[j+2]-sortposition[j+1] <= 4 && sortposition[j+1]-sortposition[j]<=4){
    let textfield  = Math.ceil((sortposition[j+2]+sortposition[j+1]+sortposition[j+0])/3)
    tooltip_array[3]+=("All layers learned at the same time at epoch " + textfield+".<br>")
  }
}
//console.log("one max?")
console.log(tooltip_array)
self.postMessage([id, "tooltip", tooltip_array])

}

async function showheatmap(id, input, firstLayer, secondLayer, classifier){
  let data_weights = []; 
  let all_weights = [input, firstLayer, secondLayer]
  let layer_name =""
  let max_node_weight  = 0 
  data_weights_array = [[],[],[]]
  
  for (let layer = 0; layer < 3; layer++){
    let number_neurons = all_weights[layer].length;
    for (let neuron = 0; neuron < number_neurons; neuron++){
      let weights = all_weights[layer][neuron].length;
      let weight_sum = 0;
      for (let weight = 0; weight < weights; weight++){
        weight_sum += Math.abs(all_weights[layer][neuron][weight]);
      } 
      if (weight_sum > max_node_weight){
        max_node_weight = weight_sum
      }
    } 
  }
  for (let layer = 0; layer < 3; layer++){
    if (layer == 0){ layer_name = "Input Layer"}
    else if (layer == 1){layer_name ="First Layer"}
    else if (layer == 2){layer_name ="Second Layer"}
    let number_neurons = all_weights[layer].length;
    for (let neuron = 0; neuron < number_neurons; neuron++){
      let weights = all_weights[layer][neuron].length;
      let weight_sum = 0;
      for (let weight = 0; weight < weights; weight++){
        weight_sum += Math.abs(all_weights[layer][neuron][weight]);
      }
      let scaled_weight_sum = weight_sum/max_node_weight
      let yposition = ((neuron*2+1)/(number_neurons*2));
      data_weights.push({"layer":layer_name,"yposition":yposition, "weight":scaled_weight_sum,"neuron":neuron})
      data_weights_array[layer].push(scaled_weight_sum)
    }

  }
  if (classifier){
  data_weights.push({"layer":"Output Layer","yposition":0.25, "weight":0.7,"neuron":0});
  data_weights.push({"layer":"Output Layer","yposition":0.75, "weight":0.7,"neuron":1});}
  else{
    data_weights.push({"layer":"Output Layer","yposition":0.5, "weight":0.7,"neuron":0});
  }
  return data_weights
  
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
  
  return [layer_input < 0.7?layer_input:0.7,layer_H1< 0.7?layer_H1:0.7,layer_H2< 0.7?layer_H2:0.7]

}

async function similarity(id, layers){
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

function pushWeightsLayer1(firstLayer,epoch){
  for (let i = 0; i< firstLayer.length; i++){
    let neuronName = "Neuron "+i
    let sum = 0
    for (let j = 0; j < firstLayer[i].length; j++){
      sum += Math.abs(firstLayer[i][j])
    }
    weights_layer1.push({epoch:epoch, neuron:neuronName, value:sum})
    weights_layer1_array[i].push(sum)
  }
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
    self.postMessage([id, "done"])
  }