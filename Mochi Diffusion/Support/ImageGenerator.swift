//
//  ImageGenerator.swift
//  Mochi Diffusion
//
//  Created by Joshua Park on 2/12/23.
//

import CoreML
@preconcurrency import GuernikaKit
@preconcurrency import StableDiffusion
import UniformTypeIdentifiers

struct GenerationConfig: Sendable, Identifiable {
    let id = UUID()
    let prompt: String
    let negativePrompt: String
    let size: CGSize?
    var initImage: CGImage?
    var inpaintMask: CGImage?
    let strength: Float
    let stepCount: Int
    var seed: UInt32
    let originalStepCount: Int
    let guidanceScale: Float
    var autosaveImages: Bool
    var imageDir: String
    var imageType: String
    var numberOfImages: Int
    let model: SDModel
    var mlComputeUnit: MLComputeUnits
    var scheduler: Scheduler
    var upscaleGeneratedImages: Bool
    var controlNets: [(name: String, image: CGImage)]
    let safetyCheckerEnabled: Bool

    func pipelineHash() -> Int {
        var hasher = Hasher()
        hasher.combine(model)
        hasher.combine(controlNets.map { $0.name })
        hasher.combine(controlNets.map { $0.image })
        hasher.combine(mlComputeUnit)
        hasher.combine(size)
        hasher.combine(initImage == nil)
        return hasher.finalize()
    }
}

@Observable public final class ImageGenerator {

    static let shared = ImageGenerator()

    enum GeneratorError: Error {
        case imageDirectoryNoAccess
        case modelDirectoryNoAccess
        case modelSubDirectoriesNoAccess
        case noModelsFound
        case pipelineNotAvailable
        case requestedModelNotFound
    }

    enum State: Sendable {
        case ready(String?)
        case error(String)
        case loading
        case running((step: Int, stepCount: Int)?)
    }

    private(set) var state = State.ready(nil)

    struct QueueProgress: Sendable {
        var index = 0
        var total = 0
    }

    private(set) var queueProgress = QueueProgress(index: 0, total: 0)

    private var guernikaPipeline: (any GuernikaPipelineProtocol)?
    private var sdPipeline: (any StableDiffusionPipelineProtocol)?

    private var generationStopped = false

    private(set) var lastStepGenerationElapsedTime: Double?

    private var generationStartTime: DispatchTime?

    func loadImages(imageDir: String) async throws -> ([SDImage], URL) {
        var finalImageDirURL: URL
        let fm = FileManager.default
        /// check if image autosave directory exists
        if imageDir.isEmpty {
            /// use default autosave directory
            finalImageDirURL = fm.homeDirectoryForCurrentUser
            finalImageDirURL.append(path: "MochiDiffusion/images", directoryHint: .isDirectory)
        } else {
            /// generate url from autosave directory
            finalImageDirURL = URL(fileURLWithPath: imageDir, isDirectory: true)
        }
        if !fm.fileExists(atPath: finalImageDirURL.path(percentEncoded: false)) {
            print(
                "Creating image autosave directory at: \"\(finalImageDirURL.path(percentEncoded: false))\""
            )
            try? fm.createDirectory(at: finalImageDirURL, withIntermediateDirectories: true)
        }
        let items = try fm.contentsOfDirectory(
            at: finalImageDirURL,
            includingPropertiesForKeys: nil,
            options: .skipsHiddenFiles
        )
        let imageURLs =
            items
            .filter { $0.isFileURL }
            .filter { ["png", "jpg", "jpeg", "heic"].contains($0.pathExtension) }
        var sdis: [SDImage] = []
        for url in imageURLs {
            guard let sdi = createSDImageFromURL(url) else { continue }
            sdis.append(sdi)
        }
        sdis.sort { $0.generatedDate < $1.generatedDate }
        return (sdis, finalImageDirURL)
    }

    private func controlNets(in controlNetDirectoryURL: URL) -> [SDControlNet] {
        let controlNetDirectoryPath = controlNetDirectoryURL.path(percentEncoded: false)

        guard FileManager.default.fileExists(atPath: controlNetDirectoryPath),
            let contentsOfControlNet = try? FileManager.default.contentsOfDirectory(
                atPath: controlNetDirectoryPath)
        else {
            return []
        }

        return contentsOfControlNet.compactMap {
            SDControlNet(url: controlNetDirectoryURL.appending(path: $0))
        }
    }

    func getModels(modelDirectoryURL: URL, controlNetDirectoryURL: URL) async throws -> [SDModel] {
        var models: [SDModel] = []
        let fm = FileManager.default

        do {
            let controlNet = controlNets(in: controlNetDirectoryURL)
            let subDirs = try modelDirectoryURL.subDirectories()

            models =
                subDirs
                .sorted {
                    $0.lastPathComponent.compare(
                        $1.lastPathComponent, options: [.caseInsensitive, .diacriticInsensitive])
                        == .orderedAscending
                }
                .compactMap { url in
                    let unetMetadataPath = url.appending(
                        components: "Unet.mlmodelc", "metadata.json"
                    ).path(percentEncoded: false)
                    let hasControlNet = fm.fileExists(atPath: unetMetadataPath)

                    if hasControlNet {
                        let controlNetSymLinkPath = url.appending(component: "controlnet").path(
                            percentEncoded: false)

                        if !fm.fileExists(atPath: controlNetSymLinkPath) {
                            try? fm.createSymbolicLink(
                                atPath: controlNetSymLinkPath,
                                withDestinationPath: controlNetDirectoryURL.path(
                                    percentEncoded: false))
                        }
                    }

                    return SDModel(
                        url: url, name: url.lastPathComponent,
                        controlNet: hasControlNet ? controlNet : [])
                }
        } catch {
            await updateState(.error("Could not get model subdirectories."))
            throw GeneratorError.modelSubDirectoriesNoAccess
        }
        if models.isEmpty {
            await updateState(
                .error("No models found under: \(modelDirectoryURL.path(percentEncoded: false))"))
            throw GeneratorError.noModelsFound
        }
        return models
    }

    func loadPipeline(
        model: SDModel,
        controlNet: [String] = [],
        computeUnit: MLComputeUnits,
        reduceMemory: Bool
    ) async throws {
        let fm = FileManager.default
        if !fm.fileExists(atPath: model.url.path) {
            await updateState(.error("Couldn't load \(model.name) because it doesn't exist."))
            throw GeneratorError.requestedModelNotFound
        }

        await updateState(.loading)
        let config = MLModelConfiguration()
        config.computeUnits = computeUnit

        if model.isXL {
            self.sdPipeline = try StableDiffusionXLPipeline(
                resourcesAt: model.url,
                configuration: config,
                reduceMemory: reduceMemory
            )
        } else {
            self.sdPipeline = try StableDiffusionPipeline(
                resourcesAt: model.url,
                controlNet: controlNet,
                configuration: config,
                disableSafety: true,
                reduceMemory: reduceMemory
            )
        }
    }

    func loadGuernikaPipeline(
        model: SDModel,
        controlNet: [String] = [],
        computeUnit: MLComputeUnits,
        reduceMemory: Bool
    ) async throws {
        let fm = FileManager.default
        if !fm.fileExists(atPath: model.url.path) {
            await updateState(.error("Couldn't load \(model.name) because it doesn't exist."))
            throw GeneratorError.requestedModelNotFound
        }

        await updateState(.loading)

        let modelresource = try GuernikaKit.load(at: model.url)

        switch modelresource {
        case is GuernikaXLPipeline:
            self.guernikaPipeline = modelresource as? GuernikaXLPipeline
        case is StableDiffusionXLRefinerPipeline:
            self.guernikaPipeline = modelresource as? StableDiffusionXLRefinerPipeline
        case is StableDiffusionPix2PixPipeline:
            self.guernikaPipeline = modelresource as? StableDiffusionPix2PixPipeline
        default:
            self.guernikaPipeline = modelresource as? StableDiffusionMainPipeline
        }

        self.guernikaPipeline?.reduceMemory = reduceMemory
        self.guernikaPipeline?.computeUnits = computeUnit.guernikaComputeUnit()
        await updateState(.ready(nil))
    }

    func generate(_ inputConfig: GenerationConfig) async throws {
        await updateState(.loading)
        generationStopped = false

        let config = inputConfig

        var sdi = SDImage()
        sdi.prompt = config.prompt
        sdi.negativePrompt = config.negativePrompt
        sdi.model = config.model.name
        sdi.scheduler = config.scheduler
        sdi.mlComputeUnit = config.mlComputeUnit
        sdi.steps = config.stepCount
        sdi.guidanceScale = Double(config.guidanceScale)

        var guernikaPipelineConfig: SampleInput?
        var sdPipelineConfig: PipelineConfiguration?

        if config.model.isGuernika {
            guernikaPipelineConfig = SampleInput(prompt: config.prompt)
            guernikaPipelineConfig?.negativePrompt = config.negativePrompt
            guernikaPipelineConfig?.size = config.size
            guernikaPipelineConfig?.initImage = config.initImage
            guernikaPipelineConfig?.inpaintMask = config.inpaintMask
            guernikaPipelineConfig?.strength = config.initImage != nil ? config.strength : 1.0
            guernikaPipelineConfig?.stepCount = config.stepCount
            guernikaPipelineConfig?.seed = config.seed
            guernikaPipelineConfig?.originalStepCount = config.originalStepCount
            guernikaPipelineConfig?.guidanceScale = config.guidanceScale
            guernikaPipelineConfig?.scheduler = convertScheduler(config.scheduler)

            guernikaPipeline?.conditioningInput = []
            for controlNetInput in config.controlNets {
                let model = config.model
                guard 
                    let controlNet = model.controlNet.first(where: { $0.name == controlNetInput.name })
                else {
                    print("Error matching selected ControlNet \(controlNetInput.name) to controlNets available to model \(model.name)")
                    continue
                }

                if controlNet.controltype == .controlNet {
                    guard let c = try? ControlNet(modelAt: controlNet.url) else {
                        continue
                    }
                    let cinput = ConditioningInput.init(module: c)
                    cinput.image = controlNetInput.image
                    guernikaPipeline?.conditioningInput.append(cinput)
                } else if controlNet.controltype == .t2IAdapter {
                    guard let a = try? T2IAdapter(modelAt: controlNet.url) else {
                        continue
                    }
                    let ainput = ConditioningInput.init(module: a)
                    ainput.image = controlNetInput.image
                    guernikaPipeline?.conditioningInput.append(ainput)
                }
            }
        } else {
            sdPipelineConfig = StableDiffusionPipeline.Configuration(prompt: config.prompt)
            sdPipelineConfig?.negativePrompt = config.negativePrompt
            sdPipelineConfig?.startingImage = config.initImage
            sdPipelineConfig?.strength = config.strength
            sdPipelineConfig?.stepCount = config.stepCount
            sdPipelineConfig?.seed = config.seed
            sdPipelineConfig?.guidanceScale = config.guidanceScale
            sdPipelineConfig?.disableSafety = !config.safetyCheckerEnabled
            // TODO: communicate the scheduler subset
            sdPipelineConfig?.schedulerType = config.scheduler == .pndm ? .pndmScheduler : .dpmSolverMultistepScheduler
            sdPipelineConfig?.controlNetInputs = config.controlNets.map { $0.image }
            if config.model.isXL {
                sdPipelineConfig?.encoderScaleFactor = 0.13025
                sdPipelineConfig?.decoderScaleFactor = 0.13025
                sdPipelineConfig?.schedulerTimestepSpacing = .karras
            }
        }

        for index in 0..<config.numberOfImages {
            await updateQueueProgress(
                QueueProgress(index: index, total: config.numberOfImages))
            generationStartTime = DispatchTime.now()

            var image: CGImage?
            if config.model.isGuernika, let guernikaPipelineConfig {
                image = try await generateGuernikaImage(
                    config, pipelineConfig: guernikaPipelineConfig)
            } else if !config.model.isGuernika, let sdPipelineConfig {
                image = try await generateSDImage(config: config, pipelineConfig: sdPipelineConfig)
            }

            if generationStopped {
                break
            }

            if let image {
                if config.upscaleGeneratedImages,
                    let upscaledImg = await Upscaler.shared.upscale(cgImage: image)
                {
                    sdi.image = upscaledImg
                    sdi.aspectRatio = CGFloat(
                        Double(upscaledImg.width) / Double(upscaledImg.height))
                    sdi.upscaler = "RealESRGAN"
                } else {
                    sdi.image = image
                    sdi.aspectRatio = CGFloat(Double(image.width) / Double(image.height))
                }
                sdi.id = UUID()
                sdi.seed = guernikaPipelineConfig?.seed ?? sdPipelineConfig?.seed ?? 0
                sdi.generatedDate = Date.now
                sdi.path = ""

                if config.autosaveImages && !config.imageDir.isEmpty {
                    var pathURL = URL(fileURLWithPath: config.imageDir, isDirectory: true)
                    let count = ImageStore.shared.images.endIndex + 1
                    pathURL.append(path: sdi.filenameWithoutExtension(count: count))

                    let type = UTType.fromString(config.imageType)
                    if let path = await sdi.save(pathURL, type: type) {
                        sdi.path = path.path(percentEncoded: false)
                    }
                }
                ImageStore.shared.add(sdi)

                guernikaPipelineConfig?.seed += 1
                sdPipelineConfig?.seed += 1
            }
        }

        await updateState(.ready(nil))
    }

    private func generateSDImage(config: GenerationConfig, pipelineConfig: PipelineConfiguration)
        async throws -> CGImage?
    {
        guard let pipeline = sdPipeline else {
            await updateState(.error("Pipeline is not loaded."))
            throw GeneratorError.pipelineNotAvailable
        }
        let images = try pipeline.generateImages(configuration: pipelineConfig) {
            progress in

            Task { @MainActor in
                state = .running((step: progress.step, stepCount: progress.stepCount))
                let endTime = DispatchTime.now()
                lastStepGenerationElapsedTime = Double(
                    endTime.uptimeNanoseconds - (generationStartTime?.uptimeNanoseconds ?? 0))
                generationStartTime = endTime
            }

            Task {
                if pipelineConfig.useDenoisedIntermediates,
                    let currentImage = progress.currentImages.last
                {
                    ImageStore.shared.setCurrentGenerating(image: currentImage)
                } else {
                    ImageStore.shared.setCurrentGenerating(image: nil)
                }
            }

            return !generationStopped
        }
        return images.first ?? nil
    }

    private func generateGuernikaImage(_ config: GenerationConfig, pipelineConfig: SampleInput)
        async throws -> CGImage?
    {
        guard let pipeline = guernikaPipeline else {
            await updateState(.error("Pipeline is not loaded."))
            throw GeneratorError.pipelineNotAvailable
        }

        let image = try pipeline.generateImages(input: pipelineConfig) {
            progress in

            Task { @MainActor in
                state = .running((step: progress.step, stepCount: progress.stepCount))
                let endTime = DispatchTime.now()
                lastStepGenerationElapsedTime = Double(
                    endTime.uptimeNanoseconds - (generationStartTime?.uptimeNanoseconds ?? 0))
                generationStartTime = endTime
            }

            Task {
                let currentImage = progress.currentLatentSample
                if await ImageController.shared.showHighqualityPreview {
                    ImageStore.shared.setCurrentGenerating(
                        image: try pipeline.decodeToImage(currentImage))
                } else {
                    ImageStore.shared.setCurrentGenerating(
                        image: pipeline.latentToImage(currentImage))
                }
            }

            return !generationStopped
        }
        return image
    }

    func stopGenerate() async {
        generationStopped = true
    }

    func updateState(_ state: State) async {
        Task { @MainActor in
            self.state = state
        }
    }

    private func updateQueueProgress(_ queueProgress: QueueProgress) async {
        Task { @MainActor in
            self.queueProgress = queueProgress
        }
    }
}

extension URL {
    func subDirectories() throws -> [URL] {
        guard hasDirectoryPath else { return [] }
        return try FileManager.default.contentsOfDirectory(
            at: self,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )
        .filter { $0.resolvingSymlinksInPath().hasDirectoryPath }
    }
}
