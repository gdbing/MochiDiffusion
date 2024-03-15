//
//  Tokenizer.swift
//  Mochi Diffusion
//
//  Created by Carter Lombardi on 2/3/23.
//

import Foundation
@preconcurrency import GuernikaKit

final class Tokenizer: Sendable {
    private let bpeTokenizer: BPETokenizer

    init?(modelDir: URL, isGuernika: Bool) {
        let mergesURL: URL
        let vocabURL: URL

        if isGuernika {
            mergesURL = modelDir.appendingPathComponent("TextEncoder.mlmodelc/merges.txt", conformingTo: .url)
            vocabURL = modelDir.appendingPathComponent("TextEncoder.mlmodelc/vocab.json", conformingTo: .url)
        } else {
            mergesURL = modelDir.appendingPathComponent("merges.txt", conformingTo: .url)
            vocabURL = modelDir.appendingPathComponent("vocab.json", conformingTo: .url)
        }

        do {
            try self.bpeTokenizer = BPETokenizer(mergesUrl: mergesURL, vocabularyUrl: vocabURL, addedVocabUrl: nil)
        } catch {
            return nil
        }
    }

    func countTokens(_ inString: String) -> Int {
        bpeTokenizer.tokenize(inString).0.count
    }
}
