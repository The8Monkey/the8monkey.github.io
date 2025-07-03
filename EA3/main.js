/* 
2 LSTM-Layer à 100 Units	returnSequences: true + 2× tf.layers.lstm
Softmax-Ausgabe	activation: 'softmax' im Dense-Layer
Cross-Entropy Loss	'sparseCategoricalCrossentropy'
Optimizer: Adam, LR = 0.01	tf.train.adam(0.01)
Batch-Size = 32	batchSize: 32 im fit()
Epochs: flexibel	epochs: 50 (oder mehr, je nach Verlauf)
Visuelle Überwachung	tfjs-vis mit fitCallbacks()
*/

let model = null;
const sequenceLength = 5;
let word2idx= null;
let idx2word = null;

// ---------- Daten vorbereiten ----------
async function loadAndPreprocessText(seqLen) {
    const text = loadText(); // kommt aus deiner externen Datei
    const words = text.replace(/\s+/g, " ").toLowerCase().split(" ");
    const vocab = [...new Set(words)];
    word2idx = Object.fromEntries(vocab.map((w, i) => [w, i]));
    idx2word = vocab;
    const inputSeqs = [];
    const targetSeqs = [];
    for (let i = 0; i + seqLen < words.length; ++i) {
        const seq = words.slice(i, i + seqLen).map((w) => word2idx[w]);
        const next = word2idx[words[i + seqLen]];
        inputSeqs.push(seq);
        targetSeqs.push(next);
    }
    return { words, vocab, word2idx, idx2word, inputSeqs, targetSeqs };
}

// ---------- Sequenzen erstellen ----------
function createSequences(words, word2idx, sequenceLength = 5) {
    const sequences = [];
    for (let i = 0; i < words.length - sequenceLength; i++) {
        const input = words
            .slice(i, i + sequenceLength)
            .map((w) => word2idx[w]);
        const label = word2idx[words[i + sequenceLength]];
        sequences.push({ input, label });
    }
    return sequences;
}

/* 


'glorotUniform'	Gut für tanh/softmax-Aktivierungen (Standard bei Dense) schnell
'heNormal'	Gut für ReLU-Aktivierungen schnell
'randomNormal'	Zufällige Normalverteilung schnell
'orthogonal'	Stabil, aber langsam bei großen Matrizen langsam

*/

// ---------- Modell definieren ----------
function buildModel(vocabSize, sequenceLength) {
    model = tf.sequential();

    model.add(
        tf.layers.embedding({
            inputDim: vocabSize,
            outputDim: 50,
            inputLength: sequenceLength,
        })
    );

    model.add(
        tf.layers.lstm({
            units: 100,
            returnSequences: true,
            kernelInitializer: "glorotUniform", // statt 'orthogonal'
            recurrentInitializer: "glorotUniform",
        })
    );

    model.add(
        tf.layers.lstm({
            units: 100,
            kernelInitializer: "glorotUniform", // statt 'orthogonal'
            recurrentInitializer: "glorotUniform",
        })
    );

    model.add(
        tf.layers.dense({
            units: vocabSize,
            activation: "softmax",
        })
    );

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"],
    });

    // Modellübersicht anzeigen
    tfvis.show.modelSummary({ name: "Modellübersicht", tab: "Modell" }, model);

    return model;
}

// ---------- Training + Loss-Visualisierung ----------
async function trainModel(
    model,
    inputSeqs,
    targetSeqs,
    seqLen,
    vocab,
    epochs = 50,
    batchSize = 32
) {
    const modelName = "lstm-lm";
    const xs = tf.tensor2d(inputSeqs, [inputSeqs.length, seqLen], "int32");
    const ys = tf.oneHot(tf.tensor1d(targetSeqs, "int32"), vocab.length);

    // Loss-Visualisierung
    const surface = { name: "Training", tab: "Loss" };
    const callbacks = tfvis.show.fitCallbacks(surface, ["loss", "acc"], {
        callbacks: ["onEpochEnd"],
    });

    await model.fit(xs, ys, {
        epochs,
        batchSize,
        callbacks: tfvis.show.fitCallbacks(surface, ["loss", "acc"], {
            callbacks: ["onEpochEnd"],
        }),
    });

    // Model lokal speichern
    // 1. Download als ZIP (Nutzer erhält files .json + .bin)
    await model.save(`downloads://${modelName}`);

    // Alternativ: LocalStorage
    // await model.save(`localstorage://${modelName}`);
}

// ---------- Vorhersage-Visualisierung ----------
function showPredictionDistribution(prediction, idx2word) {
    const probs = prediction.dataSync();
    const topK = Array.from(probs)
        .map((p, i) => ({ word: idx2word[i], prob: p }))
        .sort((a, b) => b.prob - a.prob)
        .slice(0, 10);

    tfvis.render.barchart(
        { name: "Top-10 Vorhersagen", tab: "Vorhersage" },
        topK.map((d) => ({ index: d.word, value: d.prob })),
        { xLabel: "Wort", yLabel: "Wahrscheinlichkeit" }
    );
}

// ---------- Textgenerierung ----------
function generateText(
    model,
    seedText,
    word2idx,
    idx2word,
    sequenceLength,
    numWords
) {
    let result = seedText.toLowerCase().split(/\s+/);
    for (let i = 0; i < numWords; i++) {
        const input = result
            .slice(-sequenceLength)
            .map((w) => word2idx[w] || 0);
        while (input.length < sequenceLength) {
            input.unshift(0); // vorne auffüllen
        }
        const inputTensor = tf.tensor2d([input]);
        const prediction = model.predict(inputTensor);
        showPredictionDistribution(prediction, idx2word); //  Visualisierung
        const predictedIdx = prediction.argMax(-1).dataSync()[0];
        result.push(idx2word[predictedIdx]);
    }
    return result.join(" ");
}

// ---------- Hauptprogramm ----------
async function main() {
    console.log(tf.getBackend());
    await tf.setBackend("webgl");
    await tf.ready();
    console.log(tf.getBackend());

    

    const { words, vocab, word2idx, idx2word, inputSeqs, targetSeqs } =
        await loadAndPreprocessText(sequenceLength);
    // const sequences = createSequences(words, word2idx, sequenceLength);
    // const sequences = createSequences(words, word2idx, sequenceLength).slice(
    //     0,
    //     2000
    // );
    model = buildModel(vocab.length, sequenceLength);
    console.log("model builded");
    await trainModel(model, inputSeqs, targetSeqs, sequenceLength, vocab);
    console.log("model trained");

    const text = generateText(
        model,
        "die sonne scheint",
        word2idx,
        idx2word,
        sequenceLength,
        10
    );
    console.log("Generierter Text:", text);
    return text;
}


const callbacks = {
    onEpochEnd: (epoch, logs) => {
        console.log(
            `Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)}, ` +
                `acc=${(logs.acc || logs.accuracy).toFixed(4)}`
        );
        tfvis.show.history(
            { name: "Verlauf", tab: "Metrics" },
            [
                { values: [logs.loss], series: ["loss"] },
                { values: [logs.acc || logs.accuracy], series: ["acc"] },
            ],
            { xLabel: "Epoch", yLabel: "" }
        );
    },
};

document.getElementById("startBtn").addEventListener("click", async () => {
    document.getElementById("generatedText").textContent =
        "Modell wird trainiert... bitte etwas Geduld";

    // Start der Hauptfunktion aus main.js
    if (typeof main === "function") {
        const result = await main();
        document.getElementById("generatedText").textContent = result;
    } else {
        console.warn("main() nicht verfügbar.");
    }
});

document.getElementById("genBtn").addEventListener("click", async () => {
    let textarea = document.getElementById("textarea");
    // console.log(text);
    
    const genText = generateText(
        model,
        textarea.value,
        word2idx,
        idx2word,
        sequenceLength,
        10
    );

    textarea.value = genText;
});