import Dashboard from "./Dashboard";

interface IndexProps {
  onLogout: () => void;
}

const Index = ({ onLogout }: IndexProps) => {
  // last login email is persisted by LoginForm
  const userEmail = localStorage.getItem("shield.userEmail") || "";
  return <Dashboard onLogout={onLogout} userEmail={userEmail} />;
};

export default Index;
