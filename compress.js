// const tf = require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs-node-gpu");
const { rbg_from_file, point_to_input, rgb_to_file } = require("./utils");
const decompress = require("./decompress");

const src_path = "sea.jpeg";

(async () => {
    let { data, info: { width, height } } = await rbg_from_file(src_path);
    rgb_to_file(data, width, height, "evolution/goal.jpeg");

    // console.log(data)
    let train_dataset = []

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            let pair = {
                input: point_to_input(x / width, y / height),
                output: [
                    data[(x + y * width) * 3 + 0],
                    data[(x + y * width) * 3 + 1],
                    data[(x + y * width) * 3 + 2]
                ]
            }
            // console.log(pair.output)
            train_dataset.push(pair)
        }
    }

    const model = tf.sequential({
        layers: [
            tf.layers.dense({ inputShape: [6], units: 50, activation: 'selu' }),
            tf.layers.dense({ units: 50, activation: 'selu' }),
            tf.layers.dense({ units: 50, activation: 'selu' }),
            tf.layers.dense({ units: 50, activation: 'sigmoid' }),
            tf.layers.dense({ units: 50, activation: 'selu' }),
            tf.layers.dense({ units: 50, activation: 'selu' }),
            tf.layers.dense({ units: 50, activation: 'selu' }),
            tf.layers.dense({ units: 50, activation: 'sigmoid' }),
            tf.layers.dense({ units: 50, activation: 'selu' }),
            tf.layers.dense({ units: 50, activation: 'selu' }),
            tf.layers.dense({ units: 50, activation: 'selu' }),
            tf.layers.dense({ units: 50, activation: 'sigmoid' }),
            tf.layers.dense({ units: 50, activation: 'selu' }),
            tf.layers.dense({ units: 50, activation: 'selu' }),
            tf.layers.dense({ units: 50, activation: 'selu' }),
            tf.layers.dense({ units: 50, activation: 'sigmoid' }),
            tf.layers.dense({ units: 50, activation: 'selu' }),
            tf.layers.dense({ units: 50, activation: 'selu' }),
            tf.layers.dense({ units: 50, activation: 'selu' }),
            tf.layers.dense({ units: 3, activation: 'sigmoid' }),
        ]
    });


    model.compile({
        loss: 'meanSquaredError', // categoricalCrossentropy
        optimizer: 'adam',
        metrics: ['MAE']
    });

    // console.log(train_dataset)

    let train_ins = tf.tensor(train_dataset.map((pair) => pair.input));
    let train_outs = tf.tensor(train_dataset.map((pair) => pair.output));



    for (let i = 0; i < 100; i++) {
        await model.fit(train_ins, train_outs, {
            epochs: 10,
            batchSize: 50000,
            shuffle: true
        })

        decompress(model, "evolution/step_" + i + ".jpeg", Math.floor(width / 3), Math.floor(height / 3), tf)
    }

    await model.save('file://compresion_model');



})()





