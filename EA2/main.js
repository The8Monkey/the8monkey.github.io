function yFunction(x) {
    return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
}

function generateData(n = 100) {
    const xValues = [];
    const yValues = [];
    for (let i = 0; i < n; i++) {
        const x = -2 + 4 * Math.random();
        const y = yFunction(x);
        xValues.push(x);
        yValues.push(y);
    }
    return { xValues, yValues };
}

function splitData(xValues, yValues) {
    const data = xValues.map((x, i) => ({ x, y: yValues[i] }));
    data.sort(() => Math.random() - 0.5);
    const trainData = data.slice(0, 50);
    const testData = data.slice(50);
    return {
        trainX: trainData.map((d) => d.x),
        trainY: trainData.map((d) => d.y),
        testX: testData.map((d) => d.x),
        testY: testData.map((d) => d.y),
    };
}

function generateGaussianNoise(mean = 0, variance = 0.05) {
    const u1 = Math.random(),
        u2 = Math.random();
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mean + Math.sqrt(variance) * z0;
}

function addNoise(yValues, noiseVariance = 0.05) {
    return yValues.map((y) => y + generateGaussianNoise(0, noiseVariance));
}

function getSortedData(xArray, yArray) {
    const data = xArray.map((x, i) => ({ x, y: yArray[i] }));
    return data.sort((a, b) => a.x - b.x);
}

function createModel() {
    const model = tf.sequential();
    model.add(
        tf.layers.dense({ units: 100, activation: "relu", inputShape: [1] })
    );
    model.add(tf.layers.dense({ units: 100, activation: "relu" }));
    model.add(tf.layers.dense({ units: 1, activation: "linear" }));
    const optimizer = tf.train.adam(0.01);
    model.compile({ optimizer: optimizer, loss: "meanSquaredError" });
    return model;
}

async function trainModel(
    model,
    trainX,
    trainY,
    testX,
    testY,
    epochs
    // callbacksArray = []
) {
    return await model.fit(trainX, trainY, {
        epochs: epochs,
        batchSize: 32,
        validationData: [testX, testY],
        // callbacks: callbacksArray.concat({
        //     onEpochEnd: async (epoch, logs) => {
        //         // console.log(
        //         //     `Epoch ${epoch + 1}/${epochs}: Loss = ${logs.loss.toFixed(
        //         //         4
        //         //     )}, Val Loss = ${logs.val_loss.toFixed(4)}`
        //         // );
        //         await tf.nextFrame();
        //     },
        // }),
    });
}

async function visualizeResults(
    modelNoNoise,
    bestFitModel,
    overFitModel,
    trainX,
    trainY,
    testX,
    testY,
    noisyTrainY,
    noisyTestY
) {
    // Für die MSE‑Berechnung als Tensoren
    const trainXTensor = tf.tensor2d(trainX, [trainX.length, 1]);
    const trainYTensor = tf.tensor2d(trainY, [trainY.length, 1]);
    const testXTensor = tf.tensor2d(testX, [testX.length, 1]);
    const testYTensor = tf.tensor2d(testY, [testY.length, 1]);
    const noisyTrainYTensor = tf.tensor2d(noisyTrainY, [noisyTrainY.length, 1]);
    const noisyTestYTensor = tf.tensor2d(noisyTestY, [noisyTestY.length, 1]);

    // R2: Visualisierung für das Modell ohne Rauschen (noise‑frei)
    visualizeCombinedChart(
        modelNoNoise,
        trainX,
        trainY,
        "r2-left",
        "Ohne Rauschen (Train) Combined"
    );
    visualizeCombinedChart(
        modelNoNoise,
        testX,
        testY,
        "r2-right",
        "Ohne Rauschen (Test) Combined"
    );

    // R3: Visualisierung für das Best‑Fit-Modell (mit Rauschen)
    visualizeCombinedChart(
        bestFitModel,
        trainX,
        noisyTrainY,
        "r3-left",
        "Best‑Fit (Train) Combined"
    );
    visualizeCombinedChart(
        bestFitModel,
        testX,
        noisyTestY,
        "r3-right",
        "Best‑Fit (Test) Combined"
    );

    // R4: Visualisierung für das Over‑Fit-Modell (mit Rauschen)
    visualizeCombinedChart(
        overFitModel,
        trainX,
        noisyTrainY,
        "r4-left",
        "Over‑Fit (Train) Combined"
    );
    visualizeCombinedChart(
        overFitModel,
        testX,
        noisyTestY,
        "r4-right",
        "Over‑Fit (Test) Combined"
    );

    // Berechnung und Anzeige des MSE für jedes Modell:
    const mseTrainNoNoise = modelNoNoise
        .evaluate(trainXTensor, trainYTensor)
        .dataSync()[0];
    const mseTestNoNoise = modelNoNoise
        .evaluate(testXTensor, testYTensor)
        .dataSync()[0];
    document.getElementById(
        "r2-left-loss"
    ).innerText = `Train MSE: ${mseTrainNoNoise.toFixed(4)}`;
    document.getElementById(
        "r2-right-loss"
    ).innerText = `Test MSE: ${mseTestNoNoise.toFixed(4)}`;

    const mseTrainBest = bestFitModel
        .evaluate(trainXTensor, noisyTrainYTensor)
        .dataSync()[0];
    const mseTestBest = bestFitModel
        .evaluate(testXTensor, noisyTestYTensor)
        .dataSync()[0];
    document.getElementById(
        "r3-left-loss"
    ).innerText = `Train MSE: ${mseTrainBest.toFixed(4)}`;
    document.getElementById(
        "r3-right-loss"
    ).innerText = `Test MSE: ${mseTestBest.toFixed(4)}`;

    const mseTrainOver = overFitModel
        .evaluate(trainXTensor, noisyTrainYTensor)
        .dataSync()[0];
    const mseTestOver = overFitModel
        .evaluate(testXTensor, noisyTestYTensor)
        .dataSync()[0];
    document.getElementById(
        "r4-left-loss"
    ).innerText = `Train MSE: ${mseTrainOver.toFixed(4)}`;
    document.getElementById(
        "r4-right-loss"
    ).innerText = `Test MSE: ${mseTestOver.toFixed(4)}`;
}

function visualizeCombinedChart(model, xValues, actualY, containerId, title) {
    // Zunächst Daten sortieren, damit kontinuierliche Linien gezeichnet werden können.
    const sortedData = getSortedData(xValues, actualY);
    const sortedX = sortedData.map((d) => d.x);

    // Berechne die tatsächlichen Werte, die Ground Truth und die Modellvorhersagen.
    const actualData = sortedData; // Die Punkte (Actual) sind bereits sortiert.
    const groundTruthData = sortedX.map((x) => ({ x: x, y: yFunction(x) }));

    const xTensor = tf.tensor2d(sortedX, [sortedX.length, 1]);
    const predTensor = model.predict(xTensor);
    const predY = Array.from(predTensor.dataSync());
    const predictionData = sortedX.map((x, i) => ({ x: x, y: predY[i] }));

    // Kombiniere alle drei Serien in einem Datenobjekt.
    const data = {
        values: [actualData, groundTruthData, predictionData],
        series: ["Actual", "Ground Truth", "Prediction"],
    };

    // Optionen: "showDots: true" soll die Punkte (Actual) anzeigen. Die Linien werden automatisch gezeichnet.
    const options = {
        xLabel: "x",
        yLabel: "y",
        height: 300,
        title: title,
        showDots: true,
    };

    tfvis.render.linechart(document.getElementById(containerId), data, options);
}

/****************************
 * Datensatz/Modell speichern, laden, testen
 ****************************/

function createCustomModel(params) {
    const model = tf.sequential();
    const inputShape = [1];
    // Erster Hidden Layer (mit inputShape)
    model.add(
        tf.layers.dense({
            units: params.unitsPerLayer || 100,
            activation: params.activation || "relu",
            inputShape: inputShape,
        })
    );
    // Weitere Hidden Layer (falls gewünscht)
    for (let i = 1; i < (params.numHiddenLayers || 2); i++) {
        model.add(
            tf.layers.dense({
                units: params.unitsPerLayer || 100,
                activation: params.activation || "relu",
            })
        );
    }
    // Output-Layer
    model.add(
        tf.layers.dense({
            units: 1,
            activation: "linear",
        })
    );
    // Optimizer mit angegebener Lernrate
    const optimizer = tf.train.adam(params.learningRate || 0.01);
    model.compile({
        optimizer: optimizer,
        loss: "meanSquaredError",
    });
    console.log("Custom Modell erstellt mit Parametern:", params);
    return model;
}

/* ---------------------------
   Datensatz als Datei speichern
------------------------------ */
function downloadDatasetAsFile(
    dataObj,
    filename = `dataset${Date.now()}.json`
) {
    if (dataObj != null) {
        const dataStr = JSON.stringify(dataObj, null, 2);
        const blob = new Blob([dataStr], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        console.log("Datensatz wurde als Datei heruntergeladen.");
        // Optionale Anzeige im UI:
        document.getElementById("dataset-status").innerText =
            "Datensatz als Datei heruntergeladen.";
    } else {
        alert("Bitte Datensatz laden bzw. erzeugen.");
        return;
    }
}

/* ---------------------------
   Datensatz über Dateiupload laden
------------------------------ */
function readDatasetFromFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const obj = JSON.parse(e.target.result);
                resolve(obj);
            } catch (error) {
                reject(error);
            }
        };
        reader.onerror = (err) => reject(err);
        reader.readAsText(file);
    });
}

function setupDatasetUploadHandler(fileInputId, callback) {
    const fileInput = document.getElementById(fileInputId);
    fileInput.addEventListener("change", async (event) => {
        const file = event.target.files[0];
        if (!file) {
            console.error("Keine Datei ausgewählt.");
            return;
        }
        try {
            const dataObj = await readDatasetFromFile(file);
            console.log("Datensatz aus Datei geladen:", dataObj);
            document.getElementById("dataset-status").innerText =
                "Datensatz aus Datei geladen.";
            callback(dataObj);
        } catch (err) {
            console.error("Fehler beim Lesen der Datensatzdatei:", err);
            document.getElementById("dataset-status").innerText =
                "Fehler beim Laden des Datensatzes.";
        }
    });
}

/* ---------------------------
   Modell als Datei speichern (Download)
------------------------------ */
async function saveTrainedModelToFile(model, modelName = "my-model") {
    try {
        const saveResult = await model.save(
            `downloads://${modelName}${Date.now()}`
        );
        console.log("Modell wurde als Datei heruntergeladen:", saveResult);
        document.getElementById(
            "model-status"
        ).innerText = `Modell "${modelName}" wurde heruntergeladen.`;
    } catch (error) {
        console.error("Fehler beim Speichern des Modells:", error);
        document.getElementById("model-status").innerText =
            "Fehler beim Speichern des Modells.";
    }
}

/* ---------------------------
   Modell über Dateiupload laden
------------------------------ */
async function loadTrainedModelFromFiles(files) {
    try {
        // Hier wird erwartet, dass "files" ein Array von File-Objekten ist.
        const model = await tf.loadLayersModel(tf.io.browserFiles(files));
        console.log("Modell wurde aus Datei geladen.");
        document.getElementById("model-status").innerText =
            "Modell wurde aus Datei geladen.";
        return model;
    } catch (err) {
        console.error("Fehler beim Laden des Modells:", err);
        document.getElementById("model-status").innerText =
            "Fehler beim Laden des Modells.";
        return null;
    }
}

function setupModelUploadHandlerAndTrain(fileInputId, callback) {
    const fileInput = document.getElementById(fileInputId);
    fileInput.addEventListener("change", async (event) => {
        let files = Array.from(event.target.files);
        if (!files || files.length === 0) {
            console.error("Keine Modelldateien ausgewählt.");
            return;
        }
        // Sortiere das Array so, dass die JSON-Datei an erster Stelle steht
        files.sort((a, b) => {
            if (a.name.endsWith(".json")) return -1;
            if (b.name.endsWith(".json")) return 1;
            return 0;
        });
        const loadedModel = await loadTrainedModelFromFiles(files);

        // Wichtig: Modell vor Trainingsbeginn erneut kompilieren
        if (loadedModel) {
            loadedModel.compile({
                optimizer: tf.train.adam(0.01),
                loss: "meanSquaredError",
            });
        }

        callback(loadedModel);

        // Direkt nach dem Laden und Kompilieren: Starte Training, wenn ein Datensatz vorhanden ist
        if (loadedModel && currentDataset) {
            // Erstelle Tensoren aus dem aktuell geladenen Datensatz
            const trainXTensor = tf.tensor2d(currentDataset.trainX, [
                currentDataset.trainX.length,
                1,
            ]);
            const trainYTensor = tf.tensor2d(currentDataset.trainY, [
                currentDataset.trainY.length,
                1,
            ]);
            const testXTensor = tf.tensor2d(currentDataset.testX, [
                currentDataset.testX.length,
                1,
            ]);
            const testYTensor = tf.tensor2d(currentDataset.testY, [
                currentDataset.testY.length,
                1,
            ]);

            document.getElementById("model-status").innerText =
                "Modell geladen und kompiliert. Starte Training...";

            // Beispiel: Trainiere für 50 weitere Epochen
            await trainModel(
                loadedModel,
                trainXTensor,
                trainYTensor,
                testXTensor,
                testYTensor,
                50
            );
            document.getElementById("model-status").innerText =
                "\nTraining abgeschlossen.";
            if (loadedModel) {
                visualizeModelPredictions(
                    loadedModel,
                    currentDataset.trainX,
                    currentDataset.noisyTrainY,
                    currentDataset.groundTruthTrain,
                    currentDataset.testX,
                    currentDataset.noisyTestY,
                    currentDataset.groundTruthTest,
                    "train-chart", // Container-ID für Trainingsvisualisierung
                    "test-chart", // Container-ID für Testvisualisierung
                    "Trained Model" // Titel-Präfix
                );
            }
        } else {
            if (!currentDataset) {
                alert(
                    "Bitte laden oder erzeugen Sie zunächst einen Datensatz."
                );
            }
        }
    });
}

async function testModel(model, testXTensor, testYTensor) {
    const evalResult = model.evaluate(testXTensor, testYTensor);
    const loss = (await evalResult.data())[0];
    console.log(`Test MSE: ${loss.toFixed(4)}`);
    document.getElementById(
        "model-status"
    ).innerText = `\nTest MSE: ${loss.toFixed(4)}`;
    return loss;
}

/****************************
 * Entwicklungskontrolle: UI Event Listener
 ****************************/

let currentDataset = null; // Zum Speichern des aktuellen Datensatzes
let currentCustomModel = null; // Aktuell trainiertes Modell

// Setze den Upload-Handler für den Datensatz
setupDatasetUploadHandler("upload-dataset-file", (dataObj) => {
    currentDataset = dataObj;
});

// Initialisiere den neuen Upload-Handler statt des alten
setupModelUploadHandlerAndTrain("upload-model-files", (model) => {
    currentCustomModel = model;
});

document.getElementById("create-dataset-btn").addEventListener("click", () => {
    // Erzeuge 100 noise‑freie Datenpunkte
    const ds = generateData(100);
    // Teile den Datensatz zufällig in Training und Test auf
    const split = splitData(ds.xValues, ds.yValues);

    // Erzeuge Rauschen für die Trainings- und Testwerte:
    // Die Noise wird hier nur auf den idealen (split.trainY / split.testY) Werten angewendet.
    const noisyTrainY = addNoise(split.trainY, 0.05);
    const noisyTestY = addNoise(split.testY, 0.05);

    // Speichere den Datensatz so, dass jeweils für Training und Test
    // die idealen Werte (Ground Truth) und die verrauschten Werte (Actual) vorliegen.
    currentDataset = {
        ...split,
        // Für Training:
        groundTruthTrain: split.trainY, // ideal
        noisyTrainY: noisyTrainY, // actual (mit Rauschen)
        // Für Test:
        groundTruthTest: split.testY, // ideal
        noisyTestY: noisyTestY, // actual (mit Rauschen)
    };

    console.log("Datensatz wurde erzeugt:");
    console.log(currentDataset);
    document.getElementById("dataset-status").innerText =
        "Datensatz wurde erzeugt (Actual: verrauscht, Ground Truth: ideal).";
});

document.getElementById("save-dataset-btn").addEventListener("click", () => {
    downloadDatasetAsFile(currentDataset);
});

document
    .getElementById("train-model-btn")
    .addEventListener("click", async () => {
        if (!currentDataset) {
            alert("Bitte zuerst einen Datensatz erzeugen!");
            return;
        }

        // Erstelle das Modell (z. B. mittels createModel oder createCustomModel)
        const model = createModel(); // Annahme: createModel() ist definiert und liefert ein tf.sequential-Modell

        // Erstelle Tensoren für das Training; verwende hier die verrauschten (actual) Werte
        const trainXTensor = tf.tensor2d(currentDataset.trainX, [
            currentDataset.trainX.length,
            1,
        ]);
        const trainYTensor = tf.tensor2d(currentDataset.noisyTrainY, [
            currentDataset.noisyTrainY.length,
            1,
        ]);
        const testXTensor = tf.tensor2d(currentDataset.testX, [
            currentDataset.testX.length,
            1,
        ]);
        const testYTensor = tf.tensor2d(currentDataset.noisyTestY, [
            currentDataset.noisyTestY.length,
            1,
        ]);

        document.getElementById("training-status").innerText =
            "Training startet...";
        await trainModel(
            model,
            trainXTensor,
            trainYTensor,
            testXTensor,
            testYTensor,
            100
        ); // Annahme: trainModel() existiert
        document.getElementById("training-status").innerText =
            "Training abgeschlossen.";

        // Anschließend visualisieren:
        // Hier verwenden wir die in currentDataset gespeicherten Daten:
        // Für Training: Actual = noisyTrainY, Ground Truth = groundTruthTrain (ideal)
        // Für Test: Actual = noisyTestY, Ground Truth = groundTruthTest
        visualizeModelPredictions(
            model,
            currentDataset.trainX,
            currentDataset.noisyTrainY,
            currentDataset.groundTruthTrain,
            currentDataset.testX,
            currentDataset.noisyTestY,
            currentDataset.groundTruthTest,
            "train-chart", // Container-ID für Trainingsvisualisierung
            "test-chart", // Container-ID für Testvisualisierung
            "Trained Model" // Titel-Präfix
        );
        currentCustomModel = model;
    });

document
    .getElementById("save-model-btn")
    .addEventListener("click", async () => {
        if (!currentCustomModel) {
            alert("Kein Modell vorhanden!");
            return;
        }
        await saveTrainedModelToFile(currentCustomModel);
    });

document
    .getElementById("test-model-btn")
    .addEventListener("click", async () => {
        if (!currentCustomModel || !currentDataset) {
            alert("Bitte Trainingsmodell und Datensatz laden bzw. erzeugen.");
            return;
        }
        const testXTensor = tf.tensor2d(currentDataset.testX, [
            currentDataset.testX.length,
            1,
        ]);
        const testYTensor = tf.tensor2d(currentDataset.testY, [
            currentDataset.testY.length,
            1,
        ]);
        await testModel(currentCustomModel, testXTensor, testYTensor);
    });

function visualizeModelPredictions(
    model,
    trainX,
    trainNoisy,
    trainGT,
    testX,
    testNoisy,
    testGT,
    containerTrainId,
    containerTestId,
    titlePrefix = "Model"
) {
    // ----- Trainingsdaten visualisieren -----
    const sortedTrainNoisy = getSortedData(trainX, trainNoisy);
    const sortedTrainGT = getSortedData(trainX, trainGT);
    const sortedTrainX = sortedTrainNoisy.map((item) => item.x);

    // Modellvorhersagen für sortierte Trainings-x-Werte
    const trainTensor = tf.tensor2d(sortedTrainX, [sortedTrainX.length, 1]);
    const trainPredTensor = model.predict(trainTensor);
    const trainPred = Array.from(trainPredTensor.dataSync());
    const trainPrediction = sortedTrainX.map((x, i) => ({
        x,
        y: trainPred[i],
    }));

    const trainData = {
        values: [sortedTrainNoisy, sortedTrainGT, trainPrediction],
        series: ["Actual (Noisy)", "Ground Truth (Ideal)", "Prediction"],
    };

    const trainOptions = {
        xLabel: "x",
        yLabel: "y",
        height: 300,
        title: `${titlePrefix} (Train)`,
        showDots: true,
    };

    tfvis.render.linechart(
        document.getElementById(containerTrainId),
        trainData,
        trainOptions
    );

    // ----- Testdaten visualisieren -----
    const sortedTestNoisy = getSortedData(testX, testNoisy);
    const sortedTestGT = getSortedData(testX, testGT);
    const sortedTestX = sortedTestNoisy.map((item) => item.x);

    const testTensor = tf.tensor2d(sortedTestX, [sortedTestX.length, 1]);
    const testPredTensor = model.predict(testTensor);
    const testPred = Array.from(testPredTensor.dataSync());
    const testPrediction = sortedTestX.map((x, i) => ({ x, y: testPred[i] }));

    const testData = {
        values: [sortedTestNoisy, sortedTestGT, testPrediction],
        series: ["Actual (Noisy)", "Ground Truth (Ideal)", "Prediction"],
    };

    const testOptions = {
        xLabel: "x",
        yLabel: "y",
        height: 300,
        title: `${titlePrefix} (Test)`,
        showDots: true,
    };

    tfvis.render.linechart(
        document.getElementById(containerTestId),
        testData,
        testOptions
    );
}

/****************************
 * Optional: Gesamtablauf inkl. Visualisierung
 ****************************/

async function runAll() {
    tf.setBackend("cpu"); // damit es schneller läuft
    await tf.ready();

    const { xValues, yValues } = generateData(100);
    const { trainX, trainY, testX, testY } = splitData(xValues, yValues);
    const trainXTensor = tf.tensor2d(trainX, [trainX.length, 1]);
    const trainYTensor = tf.tensor2d(trainY, [trainY.length, 1]);
    const testXTensor = tf.tensor2d(testX, [testX.length, 1]);
    const testYTensor = tf.tensor2d(testY, [testY.length, 1]);
    const noisyTrainY = addNoise(trainY, 0.05);
    const noisyTestY = addNoise(testY, 0.05);

    const modelNoNoise = createModel();
    await trainModel(
        modelNoNoise,
        trainXTensor,
        trainYTensor,
        testXTensor,
        testYTensor,
        100
    );

    const earlyStoppingCallback = tf.callbacks.earlyStopping({
        monitor: "val_loss",
        patience: 10,
        verbose: 1,
    });
    const bestFitModel = createModel([earlyStoppingCallback]);
    await trainModel(
        bestFitModel,
        trainXTensor,
        tf.tensor2d(noisyTrainY, [noisyTrainY.length, 1]),
        testXTensor,
        tf.tensor2d(noisyTestY, [noisyTestY.length, 1]),
        200
    );

    const overFitModel = createModel();
    await trainModel(
        overFitModel,
        trainXTensor,
        tf.tensor2d(noisyTrainY, [noisyTrainY.length, 1]),
        testXTensor,
        tf.tensor2d(noisyTestY, [noisyTestY.length, 1]),
        1000
    );

    const spinners = document.getElementsByClassName("loader");
    Array.from(spinners).forEach((spinner) => {
        spinner.style.display = "none";
    });

    visualizeResults(
        modelNoNoise,
        bestFitModel,
        overFitModel,
        trainX,
        trainY,
        testX,
        testY,
        noisyTrainY,
        noisyTestY
    );
}

runAll();
