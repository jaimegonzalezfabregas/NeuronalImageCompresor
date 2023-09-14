const sharp = require("sharp")
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

function point_to_input(x, y) {
    return [x, y, sin(x * 2 * Math.PI), sin(y * 2 * Math.PI), sin(x * 2 * Math.PI * 6), sin(y * 2 * Math.PI * 6)]
}

module.exports = {
    rbg_from_file, rgb_to_file, point_to_input
}