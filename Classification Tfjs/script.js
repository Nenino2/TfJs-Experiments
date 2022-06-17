import {TRAINING_DATA} from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js';

const INPUTS = TRAINING_DATA.inputs;
const OUTPUTS = TRAINING_DATA.outputs;

tf.util.shuffleCombo(INPUTS, OUTPUTS);

const INPUTS_TENSOR = tf.tensor2d(INPUTS);
const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, "int32"), 10);

const model = tf.sequential();

model.add(tf.layers.dense({inputShape: [784], units: 32, activation: "relu"}))
model.add(tf.layers.dense({units: 16, activation: "relu"}))
model.add(tf.layers.dense({units: 10, activation: "softmax"}))

model.summary()

async function train() {
    model.compile({
        optimizer: "adam",
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"]
    })

    let results = await model.fit(INPUTS_TENSOR, OUTPUTS_TENSOR, {
        shuffle: true,
        validationSplit: 0.2,
        batchSize: 512,
        epochs: 50,
        callbacks: {onEpochEnd: logProgress}
    })

    OUTPUTS_TENSOR.dispose()
    INPUTS_TENSOR.dispose()

    evaluate()
}

train()

function logProgress(index, data) {
    console.log("Accuracy: " , data.acc)
}

const PREDICTION_ELEMENT = document.getElementById("prediction");

async function evaluate() {
    const OFFSET = Math.floor((Math.random() * INPUTS.length))

    let answer = tf.tidy(() => {
        let newInput = tf.tensor1d(INPUTS[OFFSET])

        let output = model.predict(newInput.expandDims())        

        output.print()

        return output.squeeze().argMax()
    })

    const index = await answer.array();
    PREDICTION_ELEMENT.innerText = index
    PREDICTION_ELEMENT.setAttribute("class", (index === OUTPUTS[OFFSET]) ? "correct" : "wrong")
    answer.dispose();

    drawImage(INPUTS[OFFSET])
}

const CANVAS = document.getElementById("canvas")

function drawImage(digit) {
    digitTensor = tf.tensor(digit, [28, 28]);

    tf.browser.toPixels(digitTensor, CANVAS);

    setTimeout(evaluate, 3000)
}