// src/lib/firebase.ts
import { initializeApp, type FirebaseOptions } from "firebase/app";
import { getAuth, setPersistence, browserLocalPersistence } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyB_lxDmzlur2YjtOGeQCuzVoUf4Z2hkQOw",
  authDomain: "capstone-b22c7.firebaseapp.com",
  projectId: "capstone-b22c7",
  appId: "1:967980569103:web:2c80aa5743ad572248c97a",
  messagingSenderId: "967980569103",
  storageBucket: "capstone-b22c7.firebasestorage.app",
  measurementId: "G-06DRV5ZKHM"
} satisfies FirebaseOptions;

// Debug: Check if env vars are loaded
console.log("Firebase Config Loaded:", {
  apiKey: firebaseConfig.apiKey ? "Present" : "Missing",
  projectId: firebaseConfig.projectId,
  authDomain: firebaseConfig.authDomain,
});

if (!firebaseConfig.apiKey || !firebaseConfig.projectId) {
  console.error("Firebase configuration is incomplete. Please check your environment variables.");
}

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);

setPersistence(auth, browserLocalPersistence).catch(() => {});

export default app;
