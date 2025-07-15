class Tokenizer {
    constructor() {
        this.wordIndex = {};
        this.indexWord = {};
        this.numWords = 0;
    }

    fitOnTexts(texts) {
        const wordSet = new Set();
        texts.forEach((text) => {
            const words = text.toLowerCase().split(/\W+/);
            words.forEach((word) => {
                if (word && !wordSet.has(word)) {
                    wordSet.add(word);
                    this.wordIndex[word] = ++this.numWords;
                    this.indexWord[this.numWords] = word;
                }
            });
        });
    }

    textsToSequences(texts) {
        return texts.map((text) => {
            const words = text.toLowerCase().split(/\W+/);
            return words.map((word) => this.wordIndex[word] || 0);
        });
    }

    download(content, fileName, contentType) {
        var a = document.createElement("a");
        var file = new Blob([content], { type: contentType });
        a.href = URL.createObjectURL(file);
        a.download = fileName;
        a.click();
    }

    saveVocab(filepath) {
        const data = JSON.stringify({
            wordIndex: this.wordIndex,
            indexWord: this.indexWord,
        });
        this.download(data, filepath, "json");
    }

    loadVocab(filepath) {
        const data = fs.readFileSync(filepath, "utf8");
        const { wordIndex, indexWord } = JSON.parse(data);
        this.wordIndex = wordIndex;
        this.indexWord = indexWord;
        this.numWords = Object.keys(this.wordIndex).length;
    }

    getWord(index) {
        if (index >= 1 && index <= this.numWords) {
            return this.indexWord[index];
        }
        return null;
    }

    getIndex(word) {
        const wordLower = word.toLowerCase();
        return this.wordIndex[wordLower] || 0;
    }
}

/* 
2 LSTM-Layer à 100 Units	returnSequences: true + 2× tf.layers.lstm
Softmax-Ausgabe	activation: 'softmax' im Dense-Layer
Cross-Entropy Loss	'sparseCategoricalCrossentropy'
Optimizer: Adam, LR = 0.01	tf.train.adam(0.01)
Batch-Size = 32	batchSize: 32 im fit()
Epochs: flexibel	epochs: 50 (oder mehr, je nach Verlauf)
Visuelle Überwachung	tfjs-vis mit fitCallbacks()
*/

/* 
Parameter
Bedeutung
kleine Werte
große Werte

sequenceLength   
Länge der Input-Sequenz – wie viele Tokens das Modell 
sieht, um das nächste vorherzusagen	Sieht wenig Kontext → evtl. schlechtere Vorhersagen
Sieht viel Kontext → mehr Rechenaufwand

units          
Anzahl der LSTM-Zellen („Neuronen“) in einer Schicht                           
Weniger Lernkapazität → evtl. Underfitting
Mehr Kapazität → mehr Tiefe, aber auch Overfitting-Risiko

outputDim        
Größe des Embedding-Vektorraums (Dimension pro Token) 
Weniger Ausdrucksstärke → Tokens ähneln sich stark
Mehr semantischer Raum → höhere Differenzierung

dropout	        
Anteil der Verbindungen im LSTM, die beim Training zufällig ignoriert werden 
Modell lernt schneller, aber überfitet eher
Modell generalisiert besser, aber lernt langsamer

recurrentDropout	
Dropout innerhalb der rekursiven Verbindung im LSTM 
Wie oben, betrifft aber Zeitabhängigkeit
Zu hoch → Schwierigkeiten beim Erinnern langer Abfolgen

adam         
Lernrate für den Adam-Optimizer 
Modell lernt sehr langsam
Modell kann unstabil werden, überspringt Minima

epochs
Wie oft alle Trainingsdaten verarbeitet werden 
Geringe Epochen → Modell lernt evtl. nicht genug
Zu viele → Overfitting, wenn kein EarlyStopping

batchSize
Anzahl der Trainingsbeispiele, die auf einmal verarbeitet werden 
Kleinere Batches → bessere Generalisierung, aber langsamer
Größere Batches → schneller, aber riskanter (Overfit)

*/

let model = null;
let word2idx = null;
let idx2word = null;
const sequenceLength = 6;
const units = 400;
const outputdim = 1000;
const adam = 0.001;
const epochs = 100;
const batchSize = 16; //16
const dropout = 0.3;
const recurrentDropout = 0.2;
const l2 = 1e-5;
const clipNorm = 3.0; //3.0
const modelName = "lstm-lm";

const visualCallback = tfvis.show.fitCallbacks(
    { name: "Training", tab: "Loss" },
    ["loss", "val_loss", "acc", "val_acc"], //, "acc", "val_acc"
    { callbacks: ["onEpochEnd"] }
);

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
// function createSequences(words, word2idx, sequenceLength = 5) {
//     const sequences = [];
//     for (let i = 0; i < words.length - sequenceLength; i++) {
//         const input = words
//             .slice(i, i + sequenceLength)
//             .map((w) => word2idx[w]);
//         const label = word2idx[words[i + sequenceLength]];
//         sequences.push({ input, label });
//     }
//     return sequences;
// }

/* 


'glorotUniform'	Gut für tanh/softmax-Aktivierungen (Standard bei Dense) schnell
'heNormal'	Gut für ReLU-Aktivierungen schnell
'randomNormal'	Zufällige Normalverteilung schnell
'orthogonal'	Stabil, aber langsam bei großen Matrizen langsam

*/

// ---------- Modell definieren ----------
function buildModel(tokenCount) {
    model = tf.sequential();

    model.add(
        tf.layers.embedding({
            inputDim: tokenCount,
            outputDim: outputdim,
            inputLength: sequenceLength,
            // embeddingsInitializer: "varianceScaling",
            // dtype: "int32",
        })
    );

    model.add(
        tf.layers.lstm({
            units: units,
            returnSequences: true,
            kernelInitializer: "glorotUniform", // statt 'orthogonal'
            recurrentInitializer: "glorotUniform",
            dropout: dropout,
            recurrentDropout: recurrentDropout,
            // dtype: "int32",
            kernelRegularizer: tf.regularizers.l2({ l2: l2 }),
        })
    );

    model.add(
        tf.layers.lstm({
            units: units,
            returnSequences: false,
            kernelInitializer: "glorotUniform", // statt 'orthogonal'
            recurrentInitializer: "glorotUniform",
            dropout: dropout,
            recurrentDropout: recurrentDropout,
            // dtype: "int32",
            kernelRegularizer: tf.regularizers.l2({ l2: l2 }),
        })
    );

    model.add(
        tf.layers.dense({
            units: tokenCount,
            activation: "softmax",
            // dtype: "int32",
            kernelRegularizer: tf.regularizers.l2({ l2: l2 }),
        })
    );

    model.compile({
        optimizer: tf.train.adam(adam),
        loss: "categoricalCrossentropy", //sparseCategoricalCrossentropy categoricalCrossentropy
        metrics: ["accuracy"],
        // clipNorm: clipNorm,
    });

    // Modellübersicht anzeigen
    tfvis.show.modelSummary({ name: "Modellübersicht", tab: "Modell" }, model);

    return model;
}

// ---------- Training + Loss-Visualisierung ----------
async function trainModel(model, sequences, tokenCount) {
    const XValues = sequences.map((seq) => seq.slice(0, -1));
    const YValues = sequences.map((seq) => seq[seq.length - 1]);

    const xs = tf.tensor2d(XValues).cast("int32");
    const ys = tf.oneHot(tf.tensor1d(YValues, "int32"), tokenCount);
    // const ys = tf
    //     .tensor1d(YValues, "int32")
    //     .reshape([YValues.length])
    //     .toFloat(); // Als float32

    // Loss-Visualisierung
    // const surface = { name: "Training", tab: "Loss" };

    const history = await model.fit(xs, ys, {
        epochs,
        batchSize,
        validationSplit: 0.3, // <-- wichtig für val_loss!
        callbacks: [
            new tf.CustomCallback(visualCallback),
            // tf.callbacks.earlyStopping({
            //     monitor: "val_loss",
            //     patience: 10,
            //     restoreBestWeight: true,
            // }),
        ],
    });

    // Alternativ: LocalStorage
    console.log(history.history.loss);
    console.log(history.history.acc);
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

// const callbacks = {
//     onEpochEnd: (epoch, logs) => {
//         console.log(
//             `Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)}, ` +
//                 `acc=${(logs.acc || logs.accuracy).toFixed(4)}`
//         );
//         tfvis.show.history(
//             { name: "Verlauf", tab: "Metrics" },
//             [
//                 { values: [logs.loss], series: ["loss"] },
//                 { values: [logs.acc || logs.accuracy], series: ["acc"] },
//             ],
//             { xLabel: "Epoch", yLabel: "" }
//         );
//     },
// };

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

tfvis.visor().unbindKeys();

// Dann eigene Tastenzuweisung hinzufügen
document.addEventListener("keydown", (event) => {
    if (event.key === "#") {
        tfvis.visor().toggle();
    }
});

async function saveTokenizerMappings(wordIndex, indexWord) {
    try {
        const wordIndexJson = JSON.stringify(wordIndex, null, 2); // Pretty print for readability
        await fs.promises.writeFile(
            "tokenizer_word_index.json",
            wordIndexJson,
            "utf-8"
        );

        const indexWordJson = JSON.stringify(indexWord, null, 2);
        await fs.promises.writeFile(
            "tokenizer_index_word.json",
            indexWordJson,
            "utf-8"
        );
    } catch (err) {
        console.error("Error saving tokenizer mappings:", err);
    }
}

function createSequences(tokens, autoWordsCount) {
    const lines = [];
    for (let i = autoWordsCount; i < tokens.length; i++) {
        const seq = tokens.slice(i - autoWordsCount, i);
        const line = seq.join(" ");
        lines.push(line);
    }
    return lines;
}

async function tokenizeSequences(lines) {
    const tokenizer = new Tokenizer();
    tokenizer.fitOnTexts(lines);
    tokenizer.saveVocab("vocab.json");
    const sequences = tokenizer.textsToSequences(lines);
    const tokenCount = tokenizer.numWords + 1;

    return { sequences, tokenCount };
}

async function loadAndTokenizeText(filePath) {
    try {
        const textContent = loadText();
        const tokens = tokenizeText(textContent);
        return tokens;
    } catch (err) {
        console.error("Error reading file:", err);
        return null;
    }
}

function tokenizeText(text) {
    const tokens = text.split(/\s+/);
    const punctuationTable = new Map([
        ["!", ""],
        ["?", ""],
        [",", ""],
        [".", ""],
        [":", ""],
        [";", ""],
        ["(", ""],
        [")", ""],
        ['"', ""],
        ["'", ""],
        ["-", ""],
    ]);

    const escapeRegExp = (string) => {
        return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    };

    const filteredTokens = tokens.map((word) => {
        let translatedWord = word;
        for (const [p, replacement] of punctuationTable) {
            const escapedP = escapeRegExp(p);
            translatedWord = translatedWord.replace(
                new RegExp(escapedP, "g"),
                replacement
            );
        }
        return translatedWord;
    });

    const alphabeticTokens = filteredTokens.filter((word) =>
        word.match(/^[a-zA-Z]+$/)
    );

    return alphabeticTokens;
}

// function splitTokens(tokens, chunkSize = 8000, overlap = 100) {
//     const chunks = [];
//     for (let i = 0; i < tokens.length; i += chunkSize - overlap) {
//         const end = Math.min(tokens.length, i + chunkSize);
//         chunks.push(tokens.slice(i, end));
//     }
//     return chunks;
// }

// ---------- Hauptprogramm ----------
async function main() {
    await tf.ready();
    // await tf.setBackend("webgl");
    // await tf.setBackend("cpu");
    await tf.registerBackend("wasm");
    await tf.setBackend("wasm");
    console.log("backend: " + tf.getBackend());

    const tokens = await loadAndTokenizeText();

    console.log("tokens: ");
    console.log(tokens);

    const createdSequences = createSequences(tokens, sequenceLength + 1);

    console.log("created Sequences: ");
    console.log(createdSequences);

    const { sequences, tokenCount } = await tokenizeSequences(createdSequences);

    console.log("sequences: ");
    console.log(sequences);
    console.log("sequences length: " + sequences.length);
    console.log("tokencount: " + tokenCount);

    console.log(`sequenceLength: ${sequenceLength}`);
    console.log(`units: ${units}`);
    console.log(`outputdim: ${outputdim}`);
    console.log(`adam: ${adam}`);
    console.log(`epochs: ${epochs}`);
    console.log(`batchSize: ${batchSize}`);
    console.log(`dropout: ${dropout}`);
    console.log(`recurrentDropout: ${recurrentDropout}`);
    console.log(`clipNorm: ${clipNorm}`);

    // model = buildModel(tokenCount);

    // console.log("model builded");
    // await trainModel(model, sequences, tokenCount);
    // console.log("model trained");
    // await model.save(`downloads://${modelName}`);

    // loadAndTrainLSTM(sequences, tokenCount);
    loadAndTestLSTM(sequences, tokenCount);
}

async function loadAndTrainLSTM(sequences, tokenCount) {
    // 1. Modell laden
    const model = await tf.loadLayersModel("./model/lstm-lm.json");
    console.log("Alte Output-Units:", model.outputs[0].shape[1]);

    // 2. Modell kompilieren
    model.compile({
        optimizer: tf.train.adam(),
        loss: "meanSquaredError",
        metrics: ["accuracy"],
    });

    // 3. Rohdaten vorbereiten
    const XValues = sequences.map((seq) => seq.slice(0, -1));
    const YValues = sequences.map((seq) => seq[seq.length - 1]);

    const xs = tf.tensor2d(XValues).cast("int32");
    const ys = tf.oneHot(tf.tensor1d(YValues, "int32"), tokenCount);

    console.log("xs shape:", xs.shape);
    console.log("ys shape:", ys.shape);

    // 3. Prüfen, ob Output-Dimension passt
    const oldUnits = model.outputs[0].shape[1];
    let trainModel = model;

    if (oldUnits !== tokenCount) {
        console.log("Output stimmt nicht — baue neues Modell auf");

        // 3.1 Alle Layers außer der letzten
        const layersExceptLast = model.layers.slice(1, -1);

        // 3.2 Input des neuen Modells (unverändert)
        const input = model.inputs[0];
        let x = input;

        // 3.3 Alte Hidden-Layers übernehmen (ggf. einfrieren)
        layersExceptLast.forEach((layer) => {
            layer.trainable = false; // optional: nur neue Schicht trainieren
            x = layer.apply(x);
        });

        // 3.4 Neue Output-Schicht
        const output = tf.layers
            .dense({ units: tokenCount, activation: "softmax" })
            .apply(x);

        // 3.5 Neues Modell instanziieren
        trainModel = tf.model({ inputs: input, outputs: output });
    }

    // 4. Kompilieren (Classification → categoricalCrossentropy)
    trainModel.compile({
        optimizer: tf.train.adam(),
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"],
    });

    // 5. Training
    const history = await trainModel.fit(xs, ys, {
        epochs,
        batchSize,
        validationSplit: 0.3, // <-- wichtig für val_loss!
        shuffle: true,
        callbacks: [
            new tf.CustomCallback(visualCallback),
            // tf.callbacks.earlyStopping({
            //     monitor: "val_loss",
            //     patience: 10,
            //     restoreBestWeight: true,
            // }),
        ],
    });

    console.log(history.history.loss);
    console.log(history.history.acc);

    // 6. Modell speichern
    await trainModel.save(`downloads://${modelName}`);
    console.log("Training abgeschlossen und Modell gespeichert.");
}

async function loadAndTestLSTM(sequences, tokenCount) {
    const model = await tf.loadLayersModel("./model/lstm-lm.json");

    const XValues = sequences.map((seq) => seq.slice(0, -1));
    const YValues = sequences.map((seq) => seq[seq.length - 1]);

    const xs = tf.tensor2d(XValues).cast("int32");
    const ys = tf.oneHot(tf.tensor1d(YValues, "int32"), tokenCount);

    model.compile({
        optimizer: tf.train.adam(),
        loss: "meanSquaredError",
        metrics: ["accuracy"],
    });

    const evalResult = await model.evaluate(xs, ys, {
        batchSize: 64,
    });

    // Für ein einzelnes Loss/Metrik-Paar:
    const loss = evalResult[0].dataSync()[0];
    const acc = evalResult[1].dataSync()[0];

    console.log(
        `Evaluation – Loss: ${loss.toFixed(4)}, Accuracy: ${acc.toFixed(4)}`
    );

    const preds = model.predict(xs); // Shape [samples, tokenCount]
    const predArray = preds.arraySync(); // JavaScript-Array

    // Beispiel: vorhergesagte Klassenindizes
    const predictedClasses = predArray.map((row) =>
        row.indexOf(Math.max(...row))
    );

    console.log("Predicted classes:", predictedClasses);

    // const cm = confusionMatrix(YValues, predictedClasses, tokenCount);
    // console.table(cm);
}

// function confusionMatrix(trueLabels, predLabels, numClasses) {
//     const matrix = Array.from({ length: numClasses }, () =>
//         Array(numClasses).fill(0)
//     );
//     trueLabels.forEach((t, i) => {
//         matrix[t][predLabels[i]]++;
//     });
//     return matrix;
// }
