import Foundation
import CoreGraphics

let gray8 = Array(repeating: Array(repeating: UInt8(32), count: 512), count: 512)
let gray16 = Array(repeating: Array(repeating: UInt16(10000), count: 512), count: 512)
let red8 = Array(repeating: Array(repeating: UInt8(64), count: 512), count: 512)
let red16 = Array(repeating: Array(repeating: UInt16(20000), count: 512), count: 512)
let green8 = Array(repeating: Array(repeating: UInt8(128), count: 512), count: 512)
let green16 = Array(repeating: Array(repeating: UInt16(30000), count: 512), count: 512)
let blue8 = Array(repeating: Array(repeating: UInt8(255), count: 512), count: 512)
let blue16 = Array(repeating: Array(repeating: UInt16(40000), count: 512), count: 512)
let alpha8 = Array(repeating: Array(repeating: UInt8(200), count: 512), count: 512)
let alpha16 = Array(repeating: Array(repeating: UInt16(50000), count: 512), count: 512)
let binary = Array(repeating: Array(repeating: true, count: 512), count: 512)

do {
    let imageRecipeGray8 = ImageRecipe(width: 512, height: 512, pixelFormat: .GRAY8, provider: try .makeProviderGrayscale(from: gray8))
    let cgImageGray8 = try CGImage.createCGImage(imageRecipe: imageRecipeGray8)
    try cgImageGray8.saveCGImage(useName: "gray8", toPath: "./test", as: .png)

    let imageRecipeGray16 = ImageRecipe(width: 512, height: 512, pixelFormat: .GRAY16, provider: try .makeProviderGrayscale(from: gray16))
    let cgImageGray16 = try CGImage.createCGImage(imageRecipe: imageRecipeGray16)
    try cgImageGray16.saveCGImage(useName: "gray16", toPath: "./test", as: .png)

    let imageRecipeRGB888 = ImageRecipe(width: 512, height: 512, pixelFormat: .RGB888, provider: try .makeProviderRGB(fromR: red8, fromG: green8, fromB: blue8, fromA: nil))
    let cgImageRGB888 = try CGImage.createCGImage(imageRecipe: imageRecipeRGB888)
    try cgImageRGB888.saveCGImage(useName: "grb888", toPath: "./test", as: .png)

    let imageRecipeRGB161616 = ImageRecipe(width: 512, height: 512, pixelFormat: .RGB161616, provider: try .makeProviderRGB(fromR: red16, fromG: green16, fromB: blue16, fromA: nil))
    let cgImageRGB161616 = try CGImage.createCGImage(imageRecipe: imageRecipeRGB161616)
    try cgImageRGB161616.saveCGImage(useName: "grb161616", toPath: "./test", as: .png)

    let imageRecipeRGBA8888 = ImageRecipe(width: 512, height: 512, pixelFormat: .RGBA8888, provider: try .makeProviderRGB(fromR: red8, fromG: green8, fromB: blue8, fromA: alpha8))
    let cgImageRGBA8888 = try CGImage.createCGImage(imageRecipe: imageRecipeRGBA8888)
    try cgImageRGBA8888.saveCGImage(useName: "grba8888", toPath: "./test", as: .png)

    let imageRecipeRGBA16161616 = ImageRecipe(width: 512, height: 512, pixelFormat: .RGBA16161616, provider: try .makeProviderRGB(fromR: red16, fromG: green16, fromB: blue16, fromA: alpha16))
    let cgImageRGBA16161616 = try CGImage.createCGImage(imageRecipe: imageRecipeRGBA16161616)
    try cgImageRGBA16161616.saveCGImage(useName: "grba16161616", toPath: "./test", as: .png)

} catch let error as NSError{
    print("Error Domain: \(error.domain)")
    if let description = error.userInfo[NSLocalizedDescriptionKey] as? String {
        print("Error Description: \(description)")
    }
}

var cnnTest = CNN()