
const MODEL_PATH = 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/sqftToPropertyPrice/model.json';

let model = undefined;

async function loadModel() {

    model = await tf.loadLayersModel(MODEL_PATH);
    model.summary();

    const input = tf.tensor2d([[870]]);
    const inputBatch = tf.tensor2d([[500], [1100], [970]]);

    const result = model.predict(input);
    const resultBatch = model.predict(inputBatch);

    result.print(); // or .array()
    resultBatch.print(); // or .array()

    input.dispose();
    inputBatch.dispose();

    result.dispose();
    resultBatch.dispose();

    model.dispose();
}


loadModel();