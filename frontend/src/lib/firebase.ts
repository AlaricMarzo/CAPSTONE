// src/lib/firebase.ts
import { initializeApp, type FirebaseOptions } from "firebase/app";
import { getAuth, setPersistence, browserLocalPersistence } from "firebase/auth";

// 1) Try to read from Vite env
const envConfig: FirebaseOptions = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
};

// 2) Fallback to hard-coded config if envs are missing (production safety net)
const fallbackConfig: FirebaseOptions = {
  apiKey: "AIzaSyB_lxDmzlur2YjtOGeQCuzVoUf4Z2hkQOw",
  authDomain: "capstone-b22c7.firebaseapp.com",
  projectId: "capstone-b22c7",
  storageBucket: "capstone-b22c7.firebasestorage.app",
  messagingSenderId: "967980569103",
  appId: "1:967980569103:web:2c80aa5743ad572248c97a",
};

const firebaseConfig: FirebaseOptions = {
  apiKey: envConfig.apiKey || fallbackConfig.apiKey,
  authDomain: envConfig.authDomain || fallbackConfig.authDomain,
  projectId: envConfig.projectId || fallbackConfig.projectId,
  storageBucket: envConfig.storageBucket || fallbackConfig.storageBucket,
  messagingSenderId: envConfig.messagingSenderId || fallbackConfig.messagingSenderId,
  appId: envConfig.appId || fallbackConfig.appId,
};

// Debug
console.log("Firebase Config Loaded:", {
  apiKey: firebaseConfig.apiKey ? "Present" : "Missing",
  projectId: firebaseConfig.projectId,
  authDomain: firebaseConfig.authDomain,
});

if (!firebaseConfig.apiKey || !firebaseConfig.projectId || !firebaseConfig.authDomain) {
  console.error("Firebase configuration is STILL incomplete. Check VITE_* env vars.");
}

// 3) Initialize Firebase
const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);

// Persist login locally (browser only)
if (typeof window !== "undefined") {
  setPersistence(auth, browserLocalPersistence).catch(() => {});
}

export default app;
