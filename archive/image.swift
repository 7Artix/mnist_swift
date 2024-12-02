import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

enum ImageError: Error {
    case withoutName
    case withoutPath
    case invalidDimensions
}

class Image {
    enum ImageType {
        case binary
        case grayscale(bits: Int)
        case color(bits: Int)
    }
    
    let withAlpha: Bool
    let width: Int
    let height: Int
    let type: ImageType
    var name: String? = nil
    var savePath: String? = nil
    var pixelData: PixelData
    
    struct PixelData {
        var binary: [[Bool]]?
        var grayscale8: [[UInt8]]?
        var grayscale16: [[UInt16]]?
        var red8: [[UInt8]]?
        var red16: [[UInt16]]?
        var green8: [[UInt8]]?
        var green16: [[UInt16]]?
        var blue8: [[UInt8]]?
        var blue16: [[UInt16]]?
        var alpha8: [[UInt8]]? // Alpha 通道
        var alpha16: [[UInt16]]?
    }
    
    // 核心初始化方法
    private init(width: Int, height: Int, type: ImageType, withAlpha: Bool, pixelData: PixelData) {
        self.width = width
        self.height = height
        self.type = type
        self.withAlpha = withAlpha
        self.pixelData = pixelData
    }
    
    // MARK: - 二值图像初始化
    convenience init(binaryData: [[Bool]], alpha: [[UInt8]]? = nil) throws {
        guard binaryData.allSatisfy({ $0.count == binaryData.first?.count }) else {
            throw ImageError.invalidDimensions
        }
        
        let height = binaryData.count
        let width = binaryData.first?.count ?? 0
        let withAlpha = alpha != nil
        
        if withAlpha {
            guard alpha!.count == height, alpha!.allSatisfy({ $0.count == width }) else {
                throw ImageError.invalidDimensions
            }
        }
        
        let pixelData = PixelData(
            binary: binaryData,
            grayscale8: nil,
            grayscale16: nil,
            red8: nil,
            red16: nil,
            green8: nil,
            green16: nil,
            blue8: nil,
            blue16: nil,
            alpha8: alpha,
            alpha16: nil
        )
        
        self.init(width: width, height: height, type: .binary, withAlpha: withAlpha, pixelData: pixelData)
    }
    
    // MARK: - 灰度图像初始化（8位）
    convenience init(grayscale8Data: [[UInt8]], alpha: [[UInt8]]? = nil) throws {
        guard grayscale8Data.allSatisfy({ $0.count == grayscale8Data.first?.count }) else {
            throw ImageError.invalidDimensions
        }
        
        let height = grayscale8Data.count
        let width = grayscale8Data.first?.count ?? 0
        let withAlpha = alpha != nil
        
        if withAlpha {
            guard alpha!.count == height, alpha!.allSatisfy({ $0.count == width }) else {
                throw ImageError.invalidDimensions
            }
        }
        
        let pixelData = PixelData(
            binary: nil,
            grayscale8: grayscale8Data,
            grayscale16: nil,
            red8: nil,
            red16: nil,
            green8: nil,
            green16: nil,
            blue8: nil,
            blue16: nil,
            alpha8: alpha
        )
        
        self.init(width: width, height: height, type: .grayscale(bits: 8), withAlpha: withAlpha, pixelData: pixelData)
    }
    
    // MARK: - 灰度图像初始化（16位）
    convenience init(grayscale16Data: [[UInt16]], alpha: [[UInt16]]? = nil) throws {
        guard grayscale16Data.allSatisfy({ $0.count == grayscale16Data.first?.count }) else {
            throw ImageError.invalidDimensions
        }
        
        let height = grayscale16Data.count
        let width = grayscale16Data.first?.count ?? 0
        let withAlpha = alpha != nil
        
        if withAlpha {
            guard alpha!.count == height, alpha!.allSatisfy({ $0.count == width }) else {
                throw ImageError.invalidDimensions
            }
        }
        
        let pixelData = PixelData(
            binary: nil,
            grayscale8: nil,
            grayscale16: grayscale16Data,
            red8: nil,
            red16: nil,
            green8: nil,
            green16: nil,
            blue8: nil,
            blue16: nil,
            alpha8: nil,
            alpha16: alpha
        )
        
        self.init(width: width, height: height, type: .grayscale(bits: 16), withAlpha: withAlpha, pixelData: pixelData)
    }
    
    // MARK: - 彩色图像初始化（8位）
    convenience init(redChannel8: [[UInt8]], greenChannel8: [[UInt8]], blueChannel8: [[UInt8]], alpha: [[UInt8]]? = nil) throws {
        // 检查各通道维度一致性
        let height = redChannel8.count
        let width = redChannel8.first?.count ?? 0
        guard greenChannel8.count == height, blueChannel8.count == height, 
              greenChannel8.first?.count == width, blueChannel8.first?.count == width else {
            throw ImageError.invalidDimensions
        }
        
        let withAlpha = alpha != nil
        
        if withAlpha {
            guard alpha!.count == height, alpha!.allSatisfy({ $0.count == width }) else {
                throw ImageError.invalidDimensions
            }
        }
        
        let pixelData = PixelData(
            binary: nil,
            grayscale8: nil,
            grayscale16: nil,
            red8: redChannel8,
            red16: nil,
            green8: greenChannel8,
            green16: nil,
            blue8: blueChannel8,
            blue16: nil,
            alpha8: alpha,
            alpha16: nil
        )
        
        self.init(width: width, height: height, type: .color(bits: 8), withAlpha: withAlpha, pixelData: pixelData)
    }

    // MARK: - 彩色图像初始化（16位）
    convenience init(redChannel16: [[UInt16]], greenChannel16: [[UInt16]], blueChannel16: [[UInt16]], alpha: [[UInt16]]? = nil) throws {
        // 检查各通道维度一致性
        let height = redChannel16.count
        let width = redChannel16.first?.count ?? 0
        guard greenChannel16.count == height, blueChannel16.count == height,
              greenChannel16.first?.count == width, blueChannel16.first?.count == width else {
            throw ImageError.invalidDimensions
        }
        
        let withAlpha = alpha != nil
        
        if withAlpha {
            guard alpha!.count == height, alpha!.allSatisfy({ $0.count == width }) else {
                throw ImageError.invalidDimensions
            }
        }
        
        let pixelData = PixelData(
            binary: nil,
            grayscale8: nil,
            grayscale16: nil,
            red8: nil,
            red16: redChannel16,
            green8: nil,
            green16: greenChannel16,
            blue8: nil,
            blue16: blueChannel16,
            alpha8: nil,
            alpha16: alpha
        )
        
        self.init(width: width, height: height, type: .color(bits: 16), withAlpha: withAlpha, pixelData: pixelData)
    }

    func savePath(_ path: String) -> Void {
        self.savePath = path
    }

    func name(_ name: String) -> Void {
        self.name = name
    }

    //MARK: 保存图片
    func saveImage() throws -> Void {
        guard let savePath = self.savePath else {
            throw ImageError.withoutPath
        }
        guard let name = self.name else {
            throw ImageError.withoutName
        }
        var imagePixelsWith1Dim8bit: [UInt8] = []
        var imagePixelsWith1Dim16bit: [UInt16] = []
        var colorSpace: CGColorSpace
        var bitsPerComponent: Int = 0
        var bitmapInfo: UInt32 = 0
        switch type {
            case .binary:
                colorSpace = CGColorSpaceCreateDeviceGray()
                bitsPerComponent = 8
                imagePixelsWith1Dim8bit = (pixelData.binary ?? []).flatMap { row in
                    row.map { $0 ? UInt8(255) : UInt8(0) }
                }
                bitmapInfo = CGImageAlphaInfo.none.rawValue
            case .grayscale(let bits):
                colorSpace = CGColorSpaceCreateDeviceGray()
                if bits == 8 {
                    switch withAlpha {
                        case true:
                            guard let gray = pixelData.grayscale8 , let alpha = pixelData.alpha8 else {
                                throw ImageError.invalidDimensions
                            }
                            for y in 0..<height {
                                for x in 0..<width {
                                    imagePixelsWith1Dim8bit.append(gray[y][x])
                                    imagePixelsWith1Dim8bit.append(alpha[y][x])
                                }
                            }
                            bitsPerComponent = 16
                            bitmapInfo = CGImageAlphaInfo.last.rawValue
                        case false:
                            guard let gray = pixelData.grayscale8 else {
                                throw ImageError.invalidDimensions
                            }
                            imagePixelsWith1Dim8bit = gray.flatMap({$0})
                            bitsPerComponent = 8
                            bitmapInfo = CGImageAlphaInfo.none.rawValue
                    }
                } else if bits == 16{
                    switch withAlpha {
                        case true:
                            guard let gray = pixelData.grayscale16 , let alpha = pixelData.alpha16 else {
                                throw ImageError.invalidDimensions
                            }
                            for y in 0..<height {
                                for x in 0..<width {
                                    imagePixelsWith1Dim16bit.append(gray[y][x])
                                    imagePixelsWith1Dim16bit.append(alpha[y][x])
                                }
                            }
                            bitsPerComponent = 16
                            bitmapInfo = CGImageAlphaInfo.last.rawValue
                        case false:
                            guard let gray = pixelData.grayscale16 else {
                                throw ImageError.invalidDimensions
                            }
                            imagePixelsWith1Dim16bit = gray.flatMap({$0})
                            bitsPerComponent = 8
                            bitmapInfo = CGImageAlphaInfo.none.rawValue
                    }
                }
            case .color(let bits):
                colorSpace = CGColorSpaceCreateDeviceRGB()
                if bits == 8 {
                    switch withAlpha {
                        case true:
                            guard let red = pixelData.red8,
                                  let green = pixelData.green8,
                                  let blue = pixelData.blue8,
                                  let alpha = pixelData.alpha8 else {
                                throw ImageError.invalidDimensions
                            }
                            for y in 0..<height {
                                for x in 0..<width {
                                    imagePixelsWith1Dim8bit.append(red[y][x])
                                    imagePixelsWith1Dim8bit.append(green[y][x])
                                    imagePixelsWith1Dim8bit.append(blue[y][x])
                                    imagePixelsWith1Dim8bit.append(alpha[y][x])
                                }
                            }
                            bitsPerComponent = 32
                            bitmapInfo = CGImageAlphaInfo.last.rawValue
                        case false:
                            guard let red = pixelData.red8,
                                  let green = pixelData.green8,
                                  let blue = pixelData.blue8 else {
                                throw ImageError.invalidDimensions
                            }
                            for y in 0..<height {
                                for x in 0..<width {
                                    imagePixelsWith1Dim8bit.append(red[y][x])
                                    imagePixelsWith1Dim8bit.append(green[y][x])
                                    imagePixelsWith1Dim8bit.append(blue[y][x])
                                }
                            }
                            bitsPerComponent = 24
                            bitmapInfo = CGImageAlphaInfo.none.rawValue
                    }
                } else if bits == 16 {
                    switch withAlpha {
                        case true:
                            guard let red = pixelData.red16,
                                  let green = pixelData.green16,
                                  let blue = pixelData.blue16,
                                  let alpha = pixelData.alpha16 else {
                                throw ImageError.invalidDimensions
                            }
                            for y in 0..<height {
                                for x in 0..<width {
                                    imagePixelsWith1Dim16bit.append(red[y][x])
                                    imagePixelsWith1Dim16bit.append(green[y][x])
                                    imagePixelsWith1Dim16bit.append(blue[y][x])
                                    imagePixelsWith1Dim16bit.append(alpha[y][x])
                                }
                            }
                            bitsPerComponent = 64
                            bitmapInfo = CGImageAlphaInfo.last.rawValue
                        case false:
                            guard let red = pixelData.red16,
                                  let green = pixelData.green16,
                                  let blue = pixelData.blue16 else {
                                throw ImageError.invalidDimensions
                            }
                            for y in 0..<height {
                                for x in 0..<width {
                                    imagePixelsWith1Dim16bit.append(red[y][x])
                                    imagePixelsWith1Dim16bit.append(green[y][x])
                                    imagePixelsWith1Dim16bit.append(blue[y][x])
                                }
                            }
                            bitsPerComponent = 48
                            bitmapInfo = CGImageAlphaInfo.none.rawValue
                    }
                }
        }
        print("color space is \(colorSpace.model),\(colorSpace.model == .rgb)")
        if let context = CGContext(data: &imagePixelsWith1Dim8bit,
        width: self.width,
        height: self.height,
        bitsPerComponent: bitsPerComponent,
        bytesPerRow: (self.width * (bitsPerComponent / 8) * (withAlpha ? 2 : 1) * (colorSpace.model == .rgb ? 3 : 1)),
        space: colorSpace,
        bitmapInfo: bitmapInfo),
        let cgImage = context.makeImage() {
            let url = URL(fileURLWithPath: savePath).appendingPathComponent("\(name).png")
            if let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) {
                CGImageDestinationAddImage(destination, cgImage, nil)
                if CGImageDestinationFinalize(destination) {
                    print("Image successfully saved as \(savePath)\(name).png")
                } else {
                    print("Failed to save image")
                }
            } else {
                print("Failed to create destination")            
            } 
        } else {
            print("Failed to create CGcontext")
        }
    }
}