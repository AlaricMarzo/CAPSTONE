// src/lib/firebase.ts
import { initializeApp, type FirebaseOptions } from "firebase/app";
import { getAuth, setPersistence, browserLocalPersistence } from "firebase/auth";

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
} satisfies FirebaseOptions;

// Debug: Check if env vars are loaded
console.log("Firebase Config Values:", {
  apiKey: firebaseConfig.apiKey,
  projectId: firebaseConfig.projectId,
  authDomain: firebaseConfig.authDomain,
  appId: firebaseConfig.appId,
  messagingSenderId: firebaseConfig.messagingSenderId,
  storageBucket: firebaseConfig.storageBucket,
});

// Check if required Firebase config is present
const requiredFields = ['apiKey', 'authDomain', 'projectId'];
const missingFields = requiredFields.filter(field => !firebaseConfig[field as keyof FirebaseOptions]);

if (missingFields.length > 0) {
  console.error(`Firebase configuration is incomplete. Missing fields: ${missingFields.join(', ')}. Please check your environment variables.`);
  // Don't initialize Firebase if config is incomplete
  throw new Error(`Firebase configuration incomplete. Missing: ${missingFields.join(', ')}`);
}

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);

setPersistence(auth, browserLocalPersistence).catch(() => {});

export default app;
