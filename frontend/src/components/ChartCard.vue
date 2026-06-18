<template>
  <section class="panel chart-card">
    <header v-if="title || subtitle" class="panel-header">
      <div>
        <h2 v-if="title">{{ title }}</h2>
        <p v-if="subtitle">{{ subtitle }}</p>
      </div>
      <slot name="actions" />
    </header>
    <VChart class="chart" autoresize :option="option" />
  </section>
</template>

<script setup lang="ts">
import type { ChartOption } from "../charts";

defineProps<{
  title?: string;
  subtitle?: string;
  option: ChartOption;
}>();
</script>
