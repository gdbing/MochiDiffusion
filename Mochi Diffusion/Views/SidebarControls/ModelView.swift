//
//  ModelView.swift
//  Mochi Diffusion
//
//  Created by Joshua Park on 12/26/22.
//

import CoreML
import SwiftUI

struct ModelView: View {
    @EnvironmentObject private var controller: ImageController
    @State private var groupedModels: [String: [SDModel]] = [:]

    func loadModels(_ models: [SDModel]) {
        let groupedModels = Dictionary(grouping: models) { (model: SDModel) -> String in
            if model.name.contains("_split-einsum") {
                return model.name.components(separatedBy: "_split-einsum").first ?? model.name
            } else {
                return model.name.components(separatedBy: "_original").first ?? model.name
            }
        }
        self.groupedModels = groupedModels
    }

    var body: some View {
        Text("Model")
            .sidebarLabelFormat()
        Menu(controller.currentModel?.name ?? "") {
            ForEach(Array(groupedModels).sorted { $0.key.localizedCaseInsensitiveCompare($1.key) == .orderedAscending }, id: \.key) { key, models in
                if models.count > 1 {
                    Button {
                        if let matchedModel = models.filter({ $0.inputSize == controller.currentModel?.inputSize && $0.attention == controller.currentModel?.attention }).first {
                            controller.currentModel = matchedModel
                        } else if let matchedModel = models.filter({ $0.inputSize == controller.currentModel?.inputSize }).first {
                            controller.currentModel = matchedModel
                        } else {
                            controller.currentModel = models.first
                        }
                    } label: {
                        Menu(key) {
                            ForEach(models) { model in
                                Button(model.name) {
                                    controller.currentModel = model
                                }
                            }
                        }
                    }
                } else if let model = models.first {
                    Button(model.name) {
                        controller.currentModel = model
                    }
                }
            }
        }
        .help(controller.currentModel?.name ?? "")
        .onAppear {
            self.loadModels(controller.models)
        }
        .onChange(of: controller.models) { models in
            self.loadModels(models)
        }
    }
}

struct ModelView_Previews: PreviewProvider {
    static var previews: some View {
        ModelView()
            .environmentObject(ImageController.shared)
    }
}
