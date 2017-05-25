var net;
net = new convnetjs.Net();

var layer_defs = [];
// input layer of size 1x1x2 (all volumes are 3D)
layer_defs.push({
  type: 'input',
  out_sx: 1,
  out_sy: 1,
  out_depth: 2
});
// some fully connected layers
/*layer_defs.push({
  type: 'fc',
  num_neurons: 120,
  activation: 'relu'
});*/
layer_defs.push({
  type: 'fc',
  num_neurons: 120,
  activation: 'relu'
});
// a softmax/svm classifier predicting probabilities for the number of classes
layer_defs.push({
  type: 'svm', //softmax or svm
  num_classes: 3
});

// create a net out of it
var net = new convnetjs.Net();
net.makeLayers(layer_defs);

var nIterations = 50;

var trainingSamples = [];
var colors = [];

var currentClass = 0;

function setup() {
  createCanvas(600, 600);
  colors = [color(255, 0, 0), color(255, 255, 0), color(0, 0, 255)];
  for (var i = 0; i < 20; i++) {
    trainingSamples.push(new TrainingSample(random(width), random(150, height), round(random(2))));
  }
  train(nIterations);
}

function draw() {
  background(255);
  for (var i = 0; i < trainingSamples.length; i++) {
    trainingSamples[i].display();
  }

  var newInput = new convnetjs.Vol([mouseX / width, mouseY / height]);
  var probability_volume2 = net.forward(newInput);
  fill(0);

  var bestGuess = "";
  var highestProp = max(probability_volume2.w[0], probability_volume2.w[1], probability_volume2.w[2]);

  if (highestProp == probability_volume2.w[0]) {
    bestGuess = "prediction: 0";
  } else if (highestProp == probability_volume2.w[1]) {
    bestGuess = "prediction: 1";
  } else if (highestProp == probability_volume2.w[2]) {
    bestGuess = "prediction: 2";
  }

  text("keys: 0-2: change current class (" + currentClass + "), t: train 20 iterations, c: clear current samples n=" + trainingSamples.length +
    /*"\n\nclass 0 probability: " + nf(probability_volume2.w[0], 1, 2) +
    "\nclass 1 probability: " + nf(probability_volume2.w[1], 1, 2) +
    "\nclass 2 probability: " + nf(probability_volume2.w[2], 1, 2) +*/
    "\n\nclass 0 probability: " + probability_volume2.w[0] +
    "\nclass 1 probability: " + probability_volume2.w[1] +
    "\nclass 2 probability: " + probability_volume2.w[2] +
    "\n" + bestGuess, 10, 25);
}


function mouseClicked() {
  trainingSamples.push(new TrainingSample(mouseX, mouseY, currentClass));
}


function keyTyped() {

  if (key === '0') {
    currentClass = 0;
  } else if (key === '1') {
    currentClass = 1;
  } else if (key === '2') {
    currentClass = 2;
  } else if (key === 't') {
    train(nIterations);
  } else if (key === 'c') {
    clearTrainingSamples();
  }
}

function TrainingSample(_x, _y, _trainedClass) {

  this.x = _x;
  this.y = _y;
  this.trainedClass = _trainedClass;
  this.fillColor = colors[_trainedClass];

  this.display = function() {
    fill(this.fillColor);
    ellipse(this.x, this.y, 30, 30);

    push();
    fill(0);
    textAlign(CENTER, CENTER);
    text(this.trainedClass, this.x, this.y)
    pop();
  }
}

function train(iterations) {
  var trainer = new convnetjs.Trainer(net, {
    learning_rate: 0.01,
    l2_decay: 0.001
  });

  for (var j = 0; j < iterations; j++) {
    for (var i = 0; i < trainingSamples.length; i++) {
      ex = new convnetjs.Vol([trainingSamples[i].x / width, trainingSamples[i].y / height]);
      trainer.train(ex, trainingSamples[i].trainedClass);
    }
  }
}

function clearTrainingSamples() {
  trainingSamples = [];
  console.log(trainingSamples.length)
}