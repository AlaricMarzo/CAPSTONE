import { useState } from "react";
import { Sidebar } from "@/components/Layout/Sidebar";
import { HeaderNav } from "@/components/Header/HeaderNav";
import { MetricCard } from "@/components/Dashboard/MetricCard";
import { RecentAlertsCard } from "@/components/Dashboard/RecentAlertCard"
import { SalesTrendsChart } from "@/components/Dashboard/SalesTrendCard"
import { ProductTrafficCard } from "@/components/Dashboard/ProductTrafficCard";
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Clock,
  Search,
  Globe,
  Bell,
  FileText,
  Calendar,
  DollarSign,
  Package,
  ShoppingCart
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { getUserMetrics, inventoryData, purchaseData } from "@/data/mockData";

function HomePage() {
  const metrics = getUserMetrics();
  
  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Dashboard Overview</h2>
          <p className="text-muted-foreground">Real-time business insights and analytics</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Calendar className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">Last 30 Days</span>
          </div>
          <Button variant="outline" size="sm" className="shadow-soft">
            <FileText className="h-4 w-4 mr-2" />
            Export Report
          </Button>
        </div>
      </div>

      {/* Search Bar */}
      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
          <Input 
            placeholder="Search products, customers, orders..." 
            className="pl-10 shadow-soft"
          />
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Total Revenue"
          value={`$${metrics.totalRevenue.toFixed(2)}`}
          change="+12.5% from last month"
          changeType="positive"
          icon={DollarSign}
          color="success"
          className="shadow-soft hover:shadow-elegant transition-shadow"
        />
        <MetricCard
          title="Total Sales"
          value={metrics.totalSales.toString()}
          change="+8.2% from last month"
          changeType="positive"
          icon={ShoppingCart}
          color="info"
          className="shadow-soft hover:shadow-elegant transition-shadow"
        />
        <MetricCard
          title="Low Stock Items"
          value={metrics.lowStockItems.toString()}
          change={`${metrics.outOfStockItems} out of stock`}
          changeType="warning"
          icon={Package}
          color="warning"
          className="shadow-soft hover:shadow-elegant transition-shadow"
        />
        <MetricCard
          title="Avg Order Value"
          value={`$${metrics.averageOrderValue.toFixed(2)}`}
          change="+5.1% from last month"
          changeType="positive"
          icon={TrendingUp}
          color="default"
          className="shadow-soft hover:shadow-elegant transition-shadow"
        />
      </div>

      {/* Charts and Alerts */}
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <SalesTrendsChart />
        </div>
        <ProductTrafficCard />
      </div>

      {/* Real-time Alerts */}
      <RecentAlertsCard />
    </div>
  );
}

function SalesPage() {
  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      <div>
        <h2 className="text-3xl font-bold text-foreground">Sales Dashboard</h2>
        <p className="text-muted-foreground">Sales Trends, Forecasting & Product Performance</p>
      </div>
      
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Total Revenue"
          value="$684,200"
          change="+62.1% from previous period"
          changeType="positive"
          color="info"
        />
        <MetricCard
          title="Total Profit"
          value="$245,800"
          change="+4.2% from previous period"
          changeType="positive"
          color="success"
        />
        <MetricCard
          title="Total Sales"
          value="4,328"
          change="+1.3% from previous period"
          changeType="positive"
          color="warning"
        />
        <MetricCard
          title="Conversion Rate"
          value="3.2%"
          change="-0.1% from previous period"
          changeType="negative"
          color="destructive"
        />
      </div>
      
      <div className="text-center py-12 text-muted-foreground">
        <p>Sales dashboard content coming soon...</p>
      </div>
    </div>
  );
}

function InventoryPage() {
  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      <div>
        <h2 className="text-3xl font-bold text-foreground">Inventory</h2>
        <p className="text-muted-foreground">Live Stock Monitoring & Allocation Logic</p>
      </div>
      
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          title="Total Products"
          value="1,248"
          change="Same as last week"
          changeType="neutral"
          color="info"
        />
        <MetricCard
          title="Low Stock Items"
          value="42"
          change="3 items below threshold"
          changeType="warning"
          color="warning"
        />
        <MetricCard
          title="Out of Stock"
          value="18"
          change="5% inventory warning"
          changeType="negative"
          color="destructive"
        />
        <MetricCard
          title="Expiring Soon"
          value="24"
          change="Within next 30 days"
          changeType="warning"
          color="warning"
        />
      </div>
      
      <div className="text-center py-12 text-muted-foreground">
        <p>Inventory management content coming soon...</p>
      </div>
    </div>
  );
}

function ReportsPage() {
  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      <div>
        <h2 className="text-3xl font-bold text-foreground">Reports</h2>
        <p className="text-muted-foreground">Analytics, Insights & Performance Reports</p>
      </div>
      
      <div className="text-center py-12 text-muted-foreground">
        <p>Reports and analytics coming soon...</p>
      </div>
    </div>
  );
}



function SettingsPage() {
  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      <div>
        <h2 className="text-3xl font-bold text-foreground">Settings</h2>
        <p className="text-muted-foreground">System Configuration & Preferences</p>
      </div>
      
      <div className="text-center py-12 text-muted-foreground">
        <p>Settings content coming soon...</p>
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('home');

  const renderContent = () => {
    switch (activeTab) {
      case 'home':
        return <HomePage />;
      case 'sales':
        return <SalesPage />;
      case 'inventory':
        return <InventoryPage />;
      case 'reports':
        return <ReportsPage />;
      case 'settings':
        return <SettingsPage />;
      default:
        return <HomePage />;
    }
  };

  return (
    <div className="flex h-screen bg-gradient-to-br from-background via-background to-muted/20">
      <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />
      <div className="flex-1 flex flex-col overflow-hidden">
        <main className="flex-1 overflow-y-auto">
          {renderContent()}
        </main>
      </div>
    </div>
  );
}