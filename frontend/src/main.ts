import { createApp } from "vue";
import { createPinia } from "pinia";
import VChart from "vue-echarts";
import { use } from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { BarChart, LineChart, ScatterChart } from "echarts/charts";
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TitleComponent,
  ToolboxComponent,
  TooltipComponent,
} from "echarts/components";

import App from "./App.vue";
import router from "./router";
import "./styles.css";

use([
  CanvasRenderer,
  BarChart,
  LineChart,
  ScatterChart,
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TitleComponent,
  ToolboxComponent,
  TooltipComponent,
]);

createApp(App).use(createPinia()).use(router).component("VChart", VChart).mount("#app");

