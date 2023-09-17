const sharp = require("sharp");
const { get_virgin_model } = require("./config");
let sin = Math.sin;


async function rbg_from_file(file) {
    const { data, info } = await sharp(file).raw().toBuffer({ resolveWithObject: true })
    // console.log(data,info)

    return { data: [...data].map(e => e / 255), info }
}

async function rgb_to_file(rgb, width, height, file) {
    sharp(
        Buffer.from(rgb.map(
            e => Math.floor(e * 255)
        )),
        { raw: { width: width, height: height, channels: 3 } }
    ).toFile(file)
}

function point_to_input(input_x, input_y, x0, y0, xf, yf) {

    if (!x0) x0 = -1
    if (!y0) y0 = -1
    if (!xf) xf = 1
    if (!yf) yf = 1

    let dx = xf - x0;
    let dy = yf - y0;

    let x = input_x * dx + x0
    let y = input_y * dy + y0

    // console.log(input_x, input_y, "to", x, y)

    return [x, y]
}

async function clone_model(og_model, tf) {
    let new_model = get_virgin_model(tf);
    new_model.setWeights(
        og_model.getWeights()
    );

    return new_model
}

module.exports = {
    rbg_from_file, rgb_to_file, point_to_input, clone_model
}