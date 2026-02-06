import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  Button,
  Image,
  StyleSheet,
  ScrollView,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-react-native";
import { bundleResourceIO } from "@tensorflow/tfjs-react-native";
import calorieData from "../data/calorie_lookup.json";

export default function App() {
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [model, setModel] = useState(null);

  useEffect(() => {
    const loadModel = async () => {
      await tf.ready();
      const loaded = await tf.loadLayersModel(
        bundleResourceIO(
          require("./tflite/model.json"),
          require("./tflite/weights.bin")
        )
      );
      setModel(loaded);
      console.log("✅ Model loaded!");
    };
    loadModel();
  }, []);

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({ base64: true });
    if (!result.canceled) {
      setImage(result.assets[0].uri);
      predict(result.assets[0].uri);
    }
  };

  const predict = async (uri) => {
    if (!model) return;
    const response = await fetch(uri);
    const blob = await response.blob();
    const imageTensor = await tf.browser
      .fromPixels(await createImageBitmap(blob))
      .resizeBilinear([224, 224])
      .expandDims(0)
      .toFloat()
      .div(tf.scalar(255));

    const preds = model.predict(imageTensor);
    const classIndex = preds.argMax(1).dataSync()[0];
    const labels = Object.keys(calorieData);
    const dish = labels[classIndex % labels.length];
    const kcal = calorieData[dish] || 200;
    setPrediction({ dish, kcal });
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>🍱 Calorie Tracker</Text>
      <Button title="Pick an Image" onPress={pickImage} />
      {image && <Image source={{ uri: image }} style={styles.img} />}
      {prediction && (
        <View style={styles.card}>
          <Text style={styles.pred}>Dish: {prediction.dish}</Text>
          <Text>Estimated Calories: {prediction.kcal} kcal</Text>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    alignItems: "center",
    justifyContent: "center",
    padding: 20,
  },
  title: { fontSize: 26, fontWeight: "bold", marginBottom: 20 },
  img: { width: 250, height: 250, marginVertical: 15, borderRadius: 10 },
  card: {
    backgroundColor: "#f0f0f0",
    padding: 20,
    borderRadius: 12,
    marginTop: 20,
  },
  pred: { fontSize: 18, fontWeight: "600" },
});
