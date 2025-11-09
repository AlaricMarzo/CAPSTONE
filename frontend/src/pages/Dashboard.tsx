import { useState, useEffect } from "react";
import { Sidebar } from "@/components/Layout/Sidebar";
import { MetricCard } from "@/components/Dashboard/MetricCard";
import { RecentAlertsCard } from "@/components/Dashboard/RecentAlertCard";
import { SalesTrendsChart } from "@/components/Dashboard/SalesTrendCard";
import { ProductTrafficCard } from "@/components/Dashboard/ProductTrafficCard";
import {
  TrendingUp,
  DollarSign,
  Package,
  ShoppingCart,
  User,
  Search,
  Calendar,
  FileText,
  AlertCircle,
  Calculator
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { getUserMetrics } from "@/data/mockData";
import DataUploadPage from "./upload";
import ProfilePage from "./Profile";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Brush,
} from "recharts";
import ProgressBar from "@/components/ui/ProgressBar";

function HomePage({ onProfileClick, onLogout }: { onProfileClick: () => void; onLogout: () => void }) {
  const metrics = getUserMetrics();
  const [confirmOpen, setConfirmOpen] = useState(false);

  const confirmLogout = () => {
    onLogout();
    setConfirmOpen(false);
  };

  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
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

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                <User className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={onProfileClick}>
                <User className="mr-2 h-4 w-4" />
                Profile
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                className="text-destructive cursor-pointer"
                onSelect={(e) => {
                  e.preventDefault();
                  setConfirmOpen(true);
                }}
              >
                Log out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          <AlertDialog open={confirmOpen} onOpenChange={setConfirmOpen}>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Are you sure you want to log out?</AlertDialogTitle>
                <AlertDialogDescription>You will be returned to the login page.</AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction onClick={confirmLogout}>Yes, log me out</AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
          <Input placeholder="Search products, customers, orders..." className="pl-10 shadow-soft" />
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard title="Total Revenue" value={`$${metrics.totalRevenue.toFixed(2)}`} change="+12.5% from last month" changeType="positive" icon={DollarSign} color="success" className="shadow-soft hover:shadow-elegant transition-shadow" />
        <MetricCard title="Total Sales" value={metrics.totalSales.toString()} change="+8.2% from last month" changeType="positive" icon={ShoppingCart} color="info" className="shadow-soft hover:shadow-elegant transition-shadow" />
        <MetricCard title="Low Stock Items" value={metrics.lowStockItems.toString()} change={`${metrics.outOfStockItems} out of stock`} changeType="warning" icon={Package} color="warning" className="shadow-soft hover:shadow-elegant transition-shadow" />
        <MetricCard title="Avg Order Value" value={`$${metrics.averageOrderValue.toFixed(2)}`} change="+5.1% from last month" changeType="positive" icon={TrendingUp} color="default" className="shadow-soft hover:shadow-elegant transition-shadow" />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2"><SalesTrendsChart /></div>
        <ProductTrafficCard />
      </div>

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
      <div className="text-center py-12 text-muted-foreground">
        <p>Inventory management content coming soon...</p>
      </div>
    </div>
  );
}

function DescriptivePage({ data, loading, error, runningAnalysis, onRunAnalysis }: {
  data: any;
  loading: boolean;
  error: string | null;
  runningAnalysis: boolean;
  onRunAnalysis: () => void;
}) {
  const [salesStartDate, setSalesStartDate] = useState('');
  const [salesEndDate, setSalesEndDate] = useState('');
  const [sortedProductMetrics, setSortedProductMetrics] = useState(data?.productMetrics || []);

  useEffect(() => {
    setSortedProductMetrics(data?.productMetrics || []);
  }, [data?.productMetrics]);

  const filteredSalesData = data?.recentSales?.filter((item: any) => {
    if (!salesStartDate && !salesEndDate) return true;
    const itemDate = new Date(item.month);
    const start = salesStartDate ? new Date(salesStartDate) : null;
    const end = salesEndDate ? new Date(salesEndDate) : null;
    if (start && itemDate < start) return false;
    if (end && itemDate > end) return false;
    return true;
  }) || [];

  const sortProductMetrics = (ascending: boolean) => {
    const sorted = [...sortedProductMetrics].sort((a, b) =>
      ascending ? a.value - b.value : b.value - a.value
    );
    setSortedProductMetrics(sorted);
  };

  if (loading) {
    return (
      <div className="flex-1 space-y-6 p-8 pt-6">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Descriptive Analytics</h2>
          <p className="text-muted-foreground">Overview of key metrics and trends</p>
        </div>
        <div className="text-center py-12 text-muted-foreground">
          <p>Loading analytics data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex-1 space-y-6 p-8 pt-6">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Descriptive Analytics</h2>
          <p className="text-muted-foreground">Overview of key metrics and trends</p>
        </div>
        <div className="text-center py-12 text-destructive">
          <p>Error loading data: {error}</p>
        </div>
      </div>
    );
  }

  const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300'];

  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Descriptive Analytics</h2>
          <p className="text-muted-foreground">Overview of key metrics and trends</p>
        </div>
        <Button
          onClick={onRunAnalysis}
          disabled={runningAnalysis}
          className="shadow-soft"
        >
          {runningAnalysis ? "Running Analysis..." : "Run Descriptive Analysis"}
        </Button>
      </div>

      {runningAnalysis && (
        <ProgressBar label="Running Descriptive Analysis..." />
      )}

      {/* Search Bar */}
      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
          <Input placeholder="Search products, categories, alerts..." className="pl-10 shadow-soft" />
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard 
          title="Total Sales" 
          value={`$${data?.totalSales?.toFixed(2) || 0}`} 
          change={`+${data?.growthRate?.toFixed(1) || 0}%`} 
          changeType="positive" 
          icon={DollarSign} 
          color="success" 
          className="shadow-soft hover:shadow-elegant transition-shadow" 
        />
        <MetricCard 
          title="Sales Growth" 
          value={`+${data?.growthRate?.toFixed(1) || 0}%`} 
          change="from last month" 
          changeType="positive" 
          icon={TrendingUp} 
          color="info" 
          className="shadow-soft hover:shadow-elegant transition-shadow" 
        />
        <MetricCard 
          title="Alerts" 
          value={data?.alertCount?.toString() || 0} 
          change="active issues" 
          changeType="warning" 
          icon={AlertCircle} 
          color="warning" 
          className="shadow-soft hover:shadow-elegant transition-shadow" 
        />
        <MetricCard 
          title="Key Metrics" 
          value={`$${data?.keyMetricsValue?.toFixed(2) || 0}`} 
          change="top performers" 
          changeType="positive" 
          icon={Package} 
          color="default" 
          className="shadow-soft hover:shadow-elegant transition-shadow" 
        />
      </div>

      {/* Charts Grid */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Recent Sales Line Chart */}
        <div className="lg:col-span-2">
          <div className="bg-card rounded-lg p-6 shadow-soft">
            <h3 className="text-lg font-semibold mb-4">Recent Sales</h3>
            <div className="flex gap-4 mb-4">
              <div className="flex flex-col">
                <label className="text-sm font-medium mb-1">Start Date</label>
                <Input
                  type="date"
                  value={salesStartDate}
                  onChange={(e) => setSalesStartDate(e.target.value)}
                  className="w-32"
                />
              </div>
              <div className="flex flex-col">
                <label className="text-sm font-medium mb-1">End Date</label>
                <Input
                  type="date"
                  value={salesEndDate}
                  onChange={(e) => setSalesEndDate(e.target.value)}
                  className="w-32"
                />
              </div>
            </div>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={filteredSalesData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Brush dataKey="month" height={30} stroke="#8884d8" />
                  <Line type="monotone" dataKey="sales" stroke="#8884d8" name="Sales" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Lead Time Line Chart */}
        <div>
          <div className="bg-card rounded-lg p-6 shadow-soft">
            <h3 className="text-lg font-semibold mb-4">Lead Time</h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data?.leadTimeData || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="week" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="leadTime" stroke="#8884d8" name="Lead Time" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      {/* Product Metrics Bar Chart */}
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="bg-card rounded-lg p-6 shadow-soft">
          <h3 className="text-lg font-semibold mb-4">Product Metrics</h3>
          <div className="flex gap-2 mb-4">
            <Button variant="outline" size="sm" onClick={() => {
              const sorted = [...(data?.productMetrics || [])].sort((a, b) => b.value - a.value);
              console.log('Sorted descending:', sorted);
            }}>
              Sort Descending
            </Button>
            <Button variant="outline" size="sm" onClick={() => {
              const sorted = [...(data?.productMetrics || [])].sort((a, b) => a.value - b.value);
              console.log('Sorted ascending:', sorted);
            }}>
              Sort Ascending
            </Button>
          </div>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data?.productMetrics || []} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" fill="#8884d8" name="Value" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Recent Alerts */}
        <RecentAlertsCard />
      </div>
    </div>
  );
}

function PredictivePage({ data, loading, error, runningAnalysis, onRunAnalysis }: {
  data: any;
  loading: boolean;
  error: string | null;
  runningAnalysis: boolean;
  onRunAnalysis: () => void;
}) {
  const [featureSort, setFeatureSort] = useState<'asc' | 'desc'>('desc');

  useEffect(() => {
    setFeatureSort('desc');
  }, [data]);

  if (loading) {
    return (
      <div className="flex-1 space-y-6 p-8 pt-6">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Predictive Analytics</h2>
          <p className="text-muted-foreground">Forecasting, Machine Learning Models & Time Series Analysis</p>
        </div>
        <div className="text-center py-12 text-muted-foreground">
          <p>Loading predictive analytics data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex-1 space-y-6 p-8 pt-6">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Predictive Analytics</h2>
          <p className="text-muted-foreground">Forecasting, Machine Learning Models & Time Series Analysis</p>
        </div>
        <div className="text-center py-12 text-destructive">
          <p>Error loading data: {error}</p>
        </div>
      </div>
    );
  }

  // Derive KPIs for accuracy
  const totalModels = 3;
  const totalForecasts = (data?.randomForest?.forecasts?.length || 0) + (data?.xgboost?.forecasts?.length || 0) + (data?.sarima?.forecasts?.length || 0);
  const avgAccuracy = ((data?.randomForest?.metrics?.[0]?.mae || 0) + (data?.xgboost?.metrics?.[0]?.mae || 0) + (data?.sarima?.metrics?.[0]?.mae || 0)) / 3;

  // Combine forecasts for main chart (assume date alignment; take last 30 for demo)
  const combinedForecasts = [
    ...(data?.randomForest?.forecasts || []).slice(-30).map(f => ({ date: f.date, model: 'RF', actual: f.actual, predicted: f.predicted })),
    ...(data?.xgboost?.forecasts || []).slice(-30).map(f => ({ date: f.date, model: 'XGB', actual: f.actual, predicted: f.predicted })),
    ...(data?.sarima?.forecasts || []).slice(-30).map(f => ({ date: f.date, model: 'SARIMA', actual: f.actual, predicted: f.predicted, lower: f.lower_bound, upper: f.upper_bound }))
  ];

  // Average feature importance (assume common features)
  const features = [...new Set([
    ...(data?.randomForest?.features || []).map(f => f.feature),
    ...(data?.xgboost?.features || []).map(f => f.feature)
  ])];
  const avgImportance = features.map(feat => {
    const rfImp = (data?.randomForest?.features || []).find(f => f.feature === feat)?.importance || 0;
    const xgbImp = (data?.xgboost?.features || []).find(f => f.feature === feat)?.importance || 0;
    return { feature: feat, importance: (rfImp + xgbImp) / 2 };
  }).sort((a, b) => featureSort === 'desc' ? b.importance - a.importance : a.importance - b.importance).slice(0, 10);

  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Predictive Analytics</h2>
          <p className="text-muted-foreground">Forecasting, Machine Learning Models & Time Series Analysis</p>
        </div>
        <Button
          onClick={onRunAnalysis}
          disabled={runningAnalysis}
          className="shadow-soft"
        >
          {runningAnalysis ? "Running Analysis..." : "Run Predictive Analysis"}
        </Button>
      </div>

      {runningAnalysis && (
        <ProgressBar label="Running Predictive Analysis..." />
      )}

      {/* Search Bar */}
      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
          <Input placeholder="Search models, forecasts, features..." className="pl-10 shadow-soft" />
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        <MetricCard 
          title="Total Models" 
          value={totalModels.toString()} 
          change="active models" 
          changeType="positive" 
          icon={Package} 
          color="success" 
          className="shadow-soft hover:shadow-elegant transition-shadow" 
        />
        <MetricCard 
          title="Total Forecasts" 
          value={totalForecasts.toString()} 
          change="generated" 
          changeType="positive" 
          icon={TrendingUp} 
          color="info" 
          className="shadow-soft hover:shadow-elegant transition-shadow" 
        />
        <MetricCard 
          title="Average Accuracy" 
          value={`${(1 - avgAccuracy).toFixed(2)}`} 
          change="MAE score" 
          changeType="positive" 
          icon={Calculator} 
          color="warning" 
          className="shadow-soft hover:shadow-elegant transition-shadow" 
        />
      </div>

      {/* Key Charts Grid */}
      <div className="grid gap-6 lg:grid-cols-2">
        {/* Combined Forecasts Line Chart */}
        <div className="bg-card rounded-lg p-6 shadow-soft">
          <h3 className="text-lg font-semibold mb-4">Combined Model Forecasts</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={combinedForecasts}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="actual" stroke="#8884d8" name="Actual" dot={false} />
                <Line type="monotone" dataKey="predicted" stroke="#82ca9d" name="Predicted" dot={false} />
                <Line type="monotone" dataKey="lower" stroke="#ffc658" name="Lower Bound" strokeDasharray="3 3" dot={false} />
                <Line type="monotone" dataKey="upper" stroke="#ff8042" name="Upper Bound" strokeDasharray="3 3" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Feature Importance Bar Chart */}
        <div className="bg-card rounded-lg p-6 shadow-soft">
          <h3 className="text-lg font-semibold mb-4">Average Feature Importance</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={avgImportance} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="feature" type="category" />
                <Tooltip />
                <Legend />
                <Bar dataKey="importance" fill="#8884d8" name="Importance" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* SARIMA Specific Chart */}
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="bg-card rounded-lg p-6 shadow-soft">
          <h3 className="text-lg font-semibold mb-4">SARIMA Forecast with Bounds</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data?.sarima?.forecasts || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="actual" stroke="#8884d8" name="Actual" />
                <Line type="monotone" dataKey="predicted" stroke="#82ca9d" name="Predicted" />
                <Line type="monotone" dataKey="lower_bound" stroke="#ffc658" name="Lower Bound" strokeDasharray="3 3" />
                <Line type="monotone" dataKey="upper_bound" stroke="#ff8042" name="Upper Bound" strokeDasharray="3 3" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Model Summary Section */}
        <div className="bg-card rounded-lg p-6 shadow-soft">
          <h3 className="text-lg font-semibold mb-4">Model Performance Summary</h3>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>Random Forest MAE: {(data?.randomForest?.metrics?.[0]?.mae || 0).toFixed(2)}</div>
              <div>XGBoost MAE: {(data?.xgboost?.metrics?.[0]?.mae || 0).toFixed(2)}</div>
              <div>SARIMA MAE: {(data?.sarima?.metrics?.[0]?.mae || 0).toFixed(2)}</div>
              <div>Overall Avg MAE: {avgAccuracy.toFixed(2)}</div>
            </div>
            <div className="text-xs text-muted-foreground">
              Forecasts generated: {totalForecasts} | Models: {totalModels}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function PrescriptivePage({ data, loading, error, runningAnalysis, onRunAnalysis }: {
  data: any;
  loading: boolean;
  error: string | null;
  runningAnalysis: boolean;
  onRunAnalysis: () => void;
}) {
  if (loading) {
    return (
      <div className="flex-1 space-y-6 p-8 pt-6">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Prescriptive Analytics</h2>
          <p className="text-muted-foreground">Optimization, Recommendations & Decision Support</p>
        </div>
        <div className="text-center py-12 text-muted-foreground">
          <p>Loading prescriptive analytics data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex-1 space-y-6 p-8 pt-6">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Prescriptive Analytics</h2>
          <p className="text-muted-foreground">Optimization, Recommendations & Decision Support</p>
        </div>
        <div className="text-center py-12 text-destructive">
          <p>Error loading data: {error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-foreground">Prescriptive Analytics</h2>
          <p className="text-muted-foreground">Optimization, Recommendations & Decision Support</p>
        </div>
        <Button
          onClick={onRunAnalysis}
          disabled={runningAnalysis}
          className="shadow-soft"
        >
          {runningAnalysis ? "Running Analysis..." : "Run Prescriptive Analysis"}
        </Button>
      </div>

      {runningAnalysis && (
        <ProgressBar label="Running Prescriptive Analysis..." />
      )}

      {/* Search Bar */}
      <div className="flex items-center gap-4">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
          <Input placeholder="Search models, products, recommendations..." className="pl-10 shadow-soft" />
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <MetricCard 
          title="Total Models" 
          value={data?.summary?.totalModels?.toString() || 7} 
          change="active models" 
          changeType="positive" 
          icon={Package} 
          color="success" 
          className="shadow-soft hover:shadow-elegant transition-shadow" 
        />
        <MetricCard 
          title="Reorder Points" 
          value={(data?.reorderPoint?.length || 0).toString()} 
          change="calculated" 
          changeType="positive" 
          icon={TrendingUp} 
          color="info" 
          className="shadow-soft hover:shadow-elegant transition-shadow" 
        />
        <MetricCard 
          title="EOQ Calculations" 
          value={(data?.eoq?.length || 0).toString()} 
          change="optimized" 
          changeType="positive" 
          icon={Calculator} 
          color="warning" 
          className="shadow-soft hover:shadow-elegant transition-shadow" 
        />
        <MetricCard 
          title="Recommendations" 
          value={data?.summary?.hasRecommendations ? "Available" : "None"} 
          change="generated" 
          changeType="positive" 
          icon={FileText} 
          color="default" 
          className="shadow-soft hover:shadow-elegant transition-shadow" 
        />
      </div>

      {/* Key Charts Grid */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Reorder Point Bar Chart */}
        <div className="lg:col-span-2">
          <div className="bg-card rounded-lg p-6 shadow-soft">
            <h3 className="text-lg font-semibold mb-4">Reorder Points</h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data?.reorderPoint || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="medicine" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="reorder_point" fill="#8884d8" name="Reorder Point" />
                  <Bar dataKey="safety_stock" fill="#82ca9d" name="Safety Stock" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* EOQ Bar Chart */}
        <div>
          <div className="bg-card rounded-lg p-6 shadow-soft">
            <h3 className="text-lg font-semibold mb-4">EOQ</h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data?.eoq || []}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="medicine" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="eoq" fill="#8884d8" name="EOQ" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      {/* What-If Analysis Line Chart */}
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="bg-card rounded-lg p-6 shadow-soft">
          <h3 className="text-lg font-semibold mb-4">What-If Analysis</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data?.whatIfAnalysis || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="scenario" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="projected_profit" stroke="#8884d8" name="Profit" />
                <Line type="monotone" dataKey="projected_cost" stroke="#82ca9d" name="Cost" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Discount Optimization Scatter */}
        <div className="bg-card rounded-lg p-6 shadow-soft">
          <h3 className="text-lg font-semibold mb-4">Discount Optimization</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart data={data?.discountByProduct || []}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="discount_pct" name="Discount %" />
                <YAxis dataKey="profit_margin" name="Profit Margin %" />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Legend />
                <Scatter name="Products" dataKey="profit_margin" fill="#8884d8" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Recommendations */}
      <div className="bg-card rounded-lg p-6 shadow-soft">
        <h3 className="text-lg font-semibold mb-4">Recommendations</h3>
        <div className="max-h-96 overflow-y-auto">
          <pre className="whitespace-pre-wrap text-sm">{data?.recommendations || "No recommendations available."}</pre>
        </div>
      </div>
    </div>
  );
}

// Create a Dashboard component that combines all the pages
interface DashboardProps {
  onLogout: () => void;
  userEmail: string;
}

export default function Dashboard({ onLogout, userEmail }: DashboardProps) {
  const [activeTab, setActiveTab] = useState("home");
  const [showProfile, setShowProfile] = useState(false);

  // Shared state for analytics data
  const [descriptiveData, setDescriptiveData] = useState<any>(null);
  const [descriptiveLoading, setDescriptiveLoading] = useState(true);
  const [descriptiveError, setDescriptiveError] = useState<string | null>(null);
  const [descriptiveRunning, setDescriptiveRunning] = useState(false);

  const [predictiveData, setPredictiveData] = useState<any>(null);
  const [predictiveLoading, setPredictiveLoading] = useState(true);
  const [predictiveError, setPredictiveError] = useState<string | null>(null);
  const [predictiveRunning, setPredictiveRunning] = useState(false);

  const [prescriptiveData, setPrescriptiveData] = useState<any>(null);
  const [prescriptiveLoading, setPrescriptiveLoading] = useState(true);
  const [prescriptiveError, setPrescriptiveError] = useState<string | null>(null);
  const [prescriptiveRunning, setPrescriptiveRunning] = useState(false);

  const handleProfileClick = () => {
    setShowProfile(true);
    setActiveTab("profile");
  };

  const handleBackToDashboard = () => {
    setShowProfile(false);
    setActiveTab("home");
  };

  // Fetch functions
  const fetchDescriptiveData = async () => {
    try {
      setDescriptiveLoading(true);
      const response = await fetch('/api/analytics/descriptive');
      if (!response.ok) {
        throw new Error('Failed to fetch data');
      }
      const result = await response.json();
      setDescriptiveData(result.data);
      setDescriptiveError(null);
    } catch (err) {
      setDescriptiveError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setDescriptiveLoading(false);
    }
  };

  const fetchPredictiveData = async () => {
    try {
      setPredictiveLoading(true);
      const response = await fetch('/api/analytics/predictive');
      if (!response.ok) {
        throw new Error('Failed to fetch data');
      }
      const result = await response.json();
      setPredictiveData(result.data);
      setPredictiveError(null);
    } catch (err) {
      setPredictiveError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setPredictiveLoading(false);
    }
  };

  const fetchPrescriptiveData = async () => {
    try {
      setPrescriptiveLoading(true);
      const response = await fetch('/api/analytics/prescriptive');
      if (!response.ok) {
        throw new Error('Failed to fetch data');
      }
      const result = await response.json();
      setPrescriptiveData(result.data);
      setPrescriptiveError(null);
    } catch (err) {
      setPrescriptiveError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setPrescriptiveLoading(false);
    }
  };

  // Run analysis functions
  const runDescriptiveAnalysis = async () => {
    try {
      setDescriptiveRunning(true);
      const response = await fetch('/api/analytics/run-descriptive', {
        method: 'POST',
      });
      if (!response.ok) {
        throw new Error('Failed to run analysis');
      }
      const result = await response.json();
      if (result.success) {
        await fetchDescriptiveData();
      } else {
        setDescriptiveError(result.error || 'Analysis failed');
      }
    } catch (err) {
      setDescriptiveError(err instanceof Error ? err.message : 'Failed to run analysis');
    } finally {
      setDescriptiveRunning(false);
    }
  };

  const runPredictiveAnalysis = async () => {
    try {
      setPredictiveRunning(true);
      const response = await fetch('/api/analytics/run-predictive', {
        method: 'POST',
      });
      if (!response.ok) {
        throw new Error('Failed to run analysis');
      }
      const result = await response.json();
      if (result.success) {
        await fetchPredictiveData();
      } else {
        setPredictiveError(result.error || 'Analysis failed');
      }
    } catch (err) {
      setPredictiveError(err instanceof Error ? err.message : 'Failed to run analysis');
    } finally {
      setPredictiveRunning(false);
    }
  };

  const runPrescriptiveAnalysis = async () => {
    try {
      setPrescriptiveRunning(true);
      const response = await fetch('/api/analytics/run-prescriptive', {
        method: 'POST',
      });
      if (!response.ok) {
        throw new Error('Failed to run analysis');
      }
      const result = await response.json();
      if (result.success) {
        await fetchPrescriptiveData();
      } else {
        setPrescriptiveError(result.error || 'Analysis failed');
      }
    } catch (err) {
      setPrescriptiveError(err instanceof Error ? err.message : 'Failed to run analysis');
    } finally {
      setPrescriptiveRunning(false);
    }
  };

  // Initial data fetch
  useEffect(() => {
    fetchDescriptiveData();
    fetchPredictiveData();
    fetchPrescriptiveData();
  }, []);

  // Event listener for analytics updates
  useEffect(() => {
    const handleUpdate = () => {
      fetchDescriptiveData();
      fetchPredictiveData();
      fetchPrescriptiveData();
    };
    window.addEventListener('analyticsUpdated', handleUpdate);
    return () => window.removeEventListener('analyticsUpdated', handleUpdate);
  }, []);

  return (
    <div className="flex h-screen bg-background">
      <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />
      <div className="flex-1 flex flex-col overflow-hidden">
        <main className="flex-1 overflow-y-auto">
          {activeTab === "home" && <HomePage onProfileClick={handleProfileClick} onLogout={onLogout} />}
          {activeTab === "sales" && <SalesPage />}
          {activeTab === "inventory" && <InventoryPage />}
          {activeTab === "upload" && <DataUploadPage />}
          {activeTab === "descriptive" && (
            <DescriptivePage
              data={descriptiveData}
              loading={descriptiveLoading}
              error={descriptiveError}
              runningAnalysis={descriptiveRunning}
              onRunAnalysis={runDescriptiveAnalysis}
            />
          )}
          {activeTab === "predictive" && (
            <PredictivePage
              data={predictiveData}
              loading={predictiveLoading}
              error={predictiveError}
              runningAnalysis={predictiveRunning}
              onRunAnalysis={runPredictiveAnalysis}
            />
          )}
          {activeTab === "prescriptive" && (
            <PrescriptivePage
              data={prescriptiveData}
              loading={prescriptiveLoading}
              error={prescriptiveError}
              runningAnalysis={prescriptiveRunning}
              onRunAnalysis={runPrescriptiveAnalysis}
            />
          )}
          {activeTab === "profile" && <ProfilePage userEmail={userEmail} onBack={handleBackToDashboard} onLogout={onLogout} />}
        </main>
      </div>
    </div>
  );
}
