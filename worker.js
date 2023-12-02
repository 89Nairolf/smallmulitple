importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs");
var model;

async function initNetwork(model, firstLayer, secondLayer, sameWeights) {
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
    model.add(tf.layers.dense({ inputShape: [2], useBias: true, units: firstLayer, activation: "sigmoid" }));
    model.add(tf.layers.dense({ units: secondLayer, useBias: true, activation: "sigmoid" }));
    model.add(tf.layers.dense({ units: 2, useBias: true, activation: "softmax" }));
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
    model.compile({ loss: "meanSquaredError", optimizer: tf.train.adam(.01), metrics: ["accuracy"] });
    return model
  }


async function training(model, id, trainingData, outputData) {
  let acc = [];
  const startTime = Date.now();
  const h = await model.fit(trainingData, outputData, {
    epochs: 5,
    validationSplit: 0.2,
    callbacks: [{
      onEpochEnd: async (epoch, logs) => {
        acc.push({ epoch: epoch, value: logs.val_loss, type: "val_loss" });
        acc.push({ epoch: epoch, value: logs.loss, type: "loss" });
        acc.push({ epoch: epoch, value: logs.val_acc, type: "val_acc" });
        acc.push({ epoch: epoch, value: logs.acc, type: "acc" });
        self.postMessage([id, "acc", acc])
        if (epoch%10 == 0){
          showheatmap(id, model)
        }
        }

    },
      //callback
    ]
  });
  let time = Date.now() - startTime
  console.log(time);
}

async function showheatmap(id, model){
  let data_weights = []; 
  let input = await model.getWeights()[0].array()
  let firstLayer = await model.getWeights()[2].array()
  let secondLayer = await model.getWeights()[4].array()
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

  data_weights.push({"layer":"Output Layer","yposition":0.33, "weight":0.7,"neuron":0});
  data_weights.push({"layer":"Output Layer","yposition":0.67, "weight":0.7,"neuron":1});
self.postMessage([id, "heatmap", data_weights])

}



  self.onmessage = async (e) => {
    const { id, firstLayer, secondLayer, sameWeights, data} = e.data;
    const trainingData = tf.tensor2d(data.map(item => [item.x, item.y]));
    //e{} -> {} after message ? 
    const outputData = tf.tensor(data.map(item => [
      item.label == 1 ? 1 : 0,
      item.label == 2 ? 1 : 0,
    ]));
    model = await initNetwork(model, firstLayer, secondLayer, sameWeights);
    await training(model, id, trainingData, outputData)
    console.log("a, okay")
  };