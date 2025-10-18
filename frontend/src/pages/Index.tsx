import { useState } from "react";
import Dashboard from "./Dashboard";
import LoginForm from "@/components/Authentication/LoginForm"; // or "../components/Authentication/LoginForm"

const Index = () => {
  const handleLogout = () => {
    // Logout logic can be handled here if needed, but since auth is managed in App.tsx, this might be optional
  };

  return <Dashboard onLogout={handleLogout} />;
};

export default Index;
