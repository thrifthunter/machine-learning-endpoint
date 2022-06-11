const fs = require("fs");
const client = require('https');
const tf = require("@tensorflow/tfjs");
const tfNode = require("@tensorflow/tfjs-node");

const fastify = require('fastify')({
    logger: true
})



//testroute
fastify.get('/', async (request, reply) => {
    return { hello: 'world' }
})

//download-image from url
function downloadImage(url, filepath) {
    return new Promise((resolve, reject) => {
        client.get(url, (res) => {
            if (res.statusCode === 200) {
                res.pipe(fs.createWriteStream(filepath))
                    .on('error', reject)
                    .once('close', () => resolve(filepath));
            } else {
                res.resume();
                reject(new Error(`Request Failed With a Status Code: ${res.statusCode}`));

            }
        });
    });
}

// predict endpoint
fastify.get('/predict-thrift', async (request, reply) => {
    const { url } = request.query
    //load model
    const MODEL_URL = "https://storage.googleapis.com/thrift-model/layer-model/model.json";
    const model = await tf.loadLayersModel(MODEL_URL);

    const channel = 3;
    const imageSize = [224, 224]; // Image size 150x150
    const label = ['hoodies',
        'hoodies-female',
        'longsleeve',
        'shirt',
        'sweatshirt',
        'sweatshirt-female'];

    //do prediction
    const category = await downloadImage(url, `/tmp/tmp-predict-img`)
        .then(() => {
            let image = fs.readFileSync(`/tmp/tmp-predict-img`);
            image = tfNode.node.decodeImage(image, channel);
            image = tf.image.resizeBilinear(image, imageSize);
            image = image.expandDims()
            const prediction = model.predict(image);
            const index = prediction.argMax(1).arraySync()[0];
            return label[index];
        })


    return reply
        .code(200)
        .header('Content-Type', 'application/json; charset=utf-8')
        .send({
            error: false,
            message: 'Predict Succes',
            predictResult: category
        })


})

/**
 * Run the server!
 */
const start = async () => {
    try {
        await fastify.listen({ port: 8081 })
    } catch (err) {
        fastify.log.error(err)
        process.exit(1)
    }
}
start()