

const tf = require("@tensorflow/tfjs-node");
const decompress = require("./decompress");
const fs = require("fs");
const { model_path, extrapolation_path } = require("./config");
const fsExtra = require("fs-extra");
(async () => {
    // fsExtra.emptyDirSync(extrapolation_path);

    let models = fs.readdirSync(model_path);
    for (const model_name of models) {
        const model = await tf.loadLayersModel('file://' + model_path + model_name + '/model.json');
        await decompress(model, extrapolation_path + "tens_read_from_" + model_name + ".jpeg", 666, 1000, tf, -10, -10, 10, 10)
        await decompress(model, extrapolation_path + "double_from_" + model_name + ".jpeg", 666, 1000, tf, -2, -2, 2, 2)
    }

})()