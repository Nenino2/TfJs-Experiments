const INPUTS = [];
for (let n = -20; n<= 20; n++) {
    INPUTS.push(n);
}

const OUTPUTS = [];
for (let INPUT of INPUTS) {
    OUTPUTS.push(INPUT*INPUT)
}

tf.util.shuffleCombo(INPUTS, OUTPUTS);

const INPUTS_TENSOR = tf.tensor1d(INPUTS);
const OUTPUTS_TENSOR = tf.tensor1d(OUTPUTS);

function normalize(tensor, min, max) {
    const result = tf.tidy(() => {
        const MIN_VALUES = min || tf.min(tensor, 0);

        const MAX_VALUES = max || tf.max(tensor, 0);

        const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);

        const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);

        const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);

        return {NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES};
    });
    return result;
}

const FEATURE_RESULTS = normalize(INPUTS_TENSOR);

FEATURE_RESULTS.MIN_VALUES.print();
FEATURE_RESULTS.MAX_VALUES.print();
FEATURE_RESULTS.NORMALIZED_VALUES.print();


INPUTS_TENSOR.dispose();

const model = tf.sequential();

model.add(tf.layers.dense({inputShape: [1], units: 100, activation: "relu"}));
model.add(tf.layers.dense({units: 100, activation: "relu"}));
model.add(tf.layers.dense({ units: 1}));

model.summary();

const LEARNING_RATE = 0.0001;
const OPTIMIZER =  tf.train.sgd(LEARNING_RATE);

async function train() {

    model.compile({
        optimizer: OPTIMIZER,
        loss: "meanSquaredError"
    })

    let results = await model.fit(FEATURE_RESULTS.NORMALIZED_VALUES, OUTPUTS_TENSOR, {
        callbacks: {onEpochEnd: (epoch, logs) => {
            console.log(`Data for epoch ${epoch} - loss: ${Math.sqrt(logs.loss)}`)
            if (epoch == 50 || epoch == 100 || epoch == 150) {
                OPTIMIZER.setLearningRate(LEARNING_RATE / 2);
            }
        }},
        shuffle: true,
        batchSize: 2,
        epochs: 200
    })

    OUTPUTS_TENSOR.dispose();
    FEATURE_RESULTS.NORMALIZED_VALUES.dispose();

    console.log("Average error loss: "+Math.sqrt(results.history.loss[results.history.loss.length - 1]));

    evaluate();
}

async function evaluate() {
    tf.tidy(() => {
        let newInput = normalize(tf.tensor1d([-5]), FEATURE_RESULTS.MIN_VALUES, FEATURE_RESULTS.MAX_VALUES);

        let output = model.predict(newInput.NORMALIZED_VALUES);

        output.print();
    })

    FEATURE_RESULTS.MIN_VALUES.dispose();
    FEATURE_RESULTS.MAX_VALUES.dispose();

    console.log(tf.memory().numTensors)

    // await model.save("downloads://my-model");
    // await model.save("localstorage://demo/newModelName")
    // const model = await tf.LoadLayersModel(url)
}

train();