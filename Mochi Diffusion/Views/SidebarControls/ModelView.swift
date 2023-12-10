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
        Picker("", selection: $controller.currentModel) {
            ForEach(Array(groupedModels).sorted { $0.key.localizedCaseInsensitiveCompare($1.key) == .orderedAscending }, id: \.key) { key, models in
                if models.count > 1 {
                    Picker(key, selection: $controller.currentModel) {
                        ForEach(models) { model in
                            Text(model.name).tag(Optional(model))
                        }
                    }.tag({() -> SDModel? in
                        if let currentModel = controller.currentModel, models.contains(currentModel) {
                            return Optional(currentModel)
                        } else {
                            return models.first
                        }
                    }())
                } else if let model = models.first {
                    Text(model.name).tag(Optional(model))
                }
            }
        }
        .labelsHidden()
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
