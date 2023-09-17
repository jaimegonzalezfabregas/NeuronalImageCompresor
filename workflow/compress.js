// const tf = require("@tensorflow/tfjs-node");
const tf = require("@tensorflow/tfjs-node-gpu");
const { rbg_from_file, point_to_input } = require("./utils");
const decompress = require("./decompress");
const { model_path, evolution_path, src_img, get_virgin_model, evolution_path_hypersampled } = require("./config");

(async () => {

    const fsExtra = require('fs-extra');
    fsExtra.emptyDirSync(evolution_path);
    fsExtra.emptyDirSync(evolution_path_hypersampled);
    fsExtra.emptyDirSync(model_path);

    let { data, info: { width, height } } = await rbg_from_file(src_img);

    // console.log(data)


    const model = get_virgin_model(tf)


    model.compile({
        loss: 'meanSquaredError', // categoricalCrossentropy
        optimizer: 'adam',
        metrics: ['MAE']
    });

    // console.log(train_dataset)


    for (let level_of_detail = 16; level_of_detail <= 1024; level_of_detail *= 2) {

        tf.start

        let train_dataset = []

        for (let n_y = 0; n_y < 1; n_y += 1 / level_of_detail) {
            for (let n_x = 0; n_x < 1; n_x += 1 / level_of_detail) {

                let x = Math.floor(n_x * width);
                let y = Math.floor(n_y * height);

                let pair = {
                    input: point_to_input(n_x, n_y),
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

        let train_ins = tf.tensor(train_dataset.map((pair) => pair.input));
        let train_outs = tf.tensor(train_dataset.map((pair) => pair.output));

        let info;
        let step = 0;
        while (!info || info.history.loss[info.history.loss.length - 1] > 0.01) {
            info = await model.fit(train_ins, train_outs, {
                epochs: 100,
                batchSize: 25000,
                shuffle: true
            })

            console.log("loss: ", info.history.loss[info.history.loss.length - 1]);

            decompress(model, evolution_path + "lod_" + level_of_detail + "_step_" + step + ".jpeg",
                level_of_detail,
                level_of_detail,
                tf)
            decompress(model, evolution_path_hypersampled + "lod_" + level_of_detail + "_step_" + step + ".jpeg",
                Math.floor(Math.min(width, Math.max(width / 10, level_of_detail * width / 100))),
                Math.floor(Math.min(height, Math.max(height / 10, level_of_detail * height / 100))),
                tf)
            // decompress(model, evolution_path_hypersampled + "lod_" + level_of_detail + "_step_" + step + ".jpeg",
            //     width / 10,
            //     height / 10,
            //     tf)
            step++;

            let model_safe_file = model_path + "lod_" + level_of_detail

            // fsExtra.mkdirSync(model_safe_file)
            await model.save('file://' + model_safe_file);
        }

        train_ins.dispose()
        train_outs.dispose()
    }
})()
