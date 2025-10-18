import Dashboard from "./Dashboard";

interface IndexProps {
  onLogout: () => void;
}

const Index = ({ onLogout }: IndexProps) => {
  return <Dashboard onLogout={onLogout} username="admin" />;
};

export default Index;
