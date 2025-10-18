import { useState } from "react";
import Dashboard from "./Dashboard";
import LoginForm from "@/components/Authentication/LoginForm"; // or "../components/Authentication/LoginForm"

const Index = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [username, setUsername] = useState("");

  const handleLogin = (user: string) => {
    setUsername(user || "User");
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    // runs only after user confirms in the logout popout
    setIsAuthenticated(false);
    setUsername("");
  };

  if (!isAuthenticated) {
    return <LoginForm onLogin={handleLogin} />;
  }

  return <Dashboard onLogout={handleLogout} username={username} />;

};

export default Index;
