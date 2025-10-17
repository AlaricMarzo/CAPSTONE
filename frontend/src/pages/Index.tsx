import Dashboard from "./Dashboard";

const Index = () => {
  const handleLogout = () => {
    // Logout logic can be handled here if needed, but since auth is managed in App.tsx, this might be optional
  };

  return <Dashboard onLogout={handleLogout} />;
};

export default Index;
