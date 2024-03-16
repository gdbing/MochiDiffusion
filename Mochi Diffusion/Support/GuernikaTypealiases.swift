//
//  GuernikaTranslationLayer.swift
//  Mochi Diffusion
//
//  Created by Graham Bing on 2024-03-15.
//

import GuernikaKit
import CoreML

typealias GuernikaPipelineProtocol = StableDiffusionPipeline
typealias GuernikaXLPipeline = StableDiffusionXLPipeline

extension MLComputeUnits {
    func guernikaComputeUnit() -> ComputeUnits {
        switch self {
        case .all: return .all
        case .cpuAndGPU: return .cpuAndGPU
        case .cpuAndNeuralEngine: return .cpuAndNeuralEngine
        default: return .all
        }
    }
}
