#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <signal.h>
#include <stdbool.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_cycles.h>

#define PORT_ID 0
#define TX_QUEUE_ID 0
#define BURST_SIZE 32
#define DESIRED_PACKET_DELAY_US 1000

static volatile bool force_quit = false;
static uint64_t total_sent_packets = 0;
static uint64_t total_sent_bytes = 0;
static volatile bool force_quit = false;
// 初始化DPDK并配置端口
void init_dpdk() {
    int ret = rte_eal_init(0, NULL);
    if (ret < 0) {
        rte_exit(EXIT_FAILURE, "Error initializing DPDK\n");
    }

    ret = rte_eth_dev_configure(PORT_ID, 1, 1, NULL);
    if (ret < 0) {
        rte_exit(EXIT_FAILURE, "Error configuring port %d\n", PORT_ID);
    }

    ret = rte_eth_rx_queue_setup(PORT_ID, 0, BURST_SIZE, 0, NULL, rte_socket_id());
    if (ret < 0) {
        rte_exit(EXIT_FAILURE, "Error setting up RX queue for port %d\n", PORT_ID);
    }

    ret = rte_eth_tx_queue_setup(PORT_ID, TX_QUEUE_ID, BURST_SIZE, 0, NULL);
    if (ret < 0) {
        rte_exit(EXIT_FAILURE, "Error setting up TX queue for port %d\n", PORT_ID);
    }

    ret = rte_eth_dev_start(PORT_ID);
    if (ret < 0) {
        rte_exit(EXIT_FAILURE, "Error starting port %d\n", PORT_ID);
    }
}

// 发送数据包并标记时间戳
void send_packets_with_timestamps() {
    struct rte_mbuf *mbufs[BURST_SIZE];
    struct rte_mempool *mpool;
    uint64_t current_time;

    mpool = rte_pktmbuf_pool_create("mbuf_pool", BURST_SIZE, 32, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
    if (!mpool) {
        rte_exit(EXIT_FAILURE, "Error creating mbuf pool\n");
    }

    // 构建数据包缓冲区 mbufs，填充数据等
    for (int i = 0; i < BURST_SIZE; i++) {
        mbufs[i] = rte_pktmbuf_alloc(mpool);
        if (!mbufs[i]) {
            rte_exit(EXIT_FAILURE, "Error allocating mbuf\n");
        }

        // 在数据包中保存发送时间戳
        current_time = rte_get_tsc_cycles();
        tx_timestamps[i] = current_time;
        rte_pktmbuf_attach_extbuf(mbufs[i], &tx_timestamps[i], sizeof(uint64_t));
    }

    // 发送数据包
    int num_tx = rte_eth_tx_burst(PORT_ID, TX_QUEUE_ID, mbufs, BURST_SIZE);
    if (num_tx > 0) {
        printf("Sent %d packets with timestamps\n", num_tx);
    int num_tx = rte_eth_tx_burst(PORT_ID, TX_QUEUE_ID, mbufs, BURST_SIZE);
    if (num_tx > 0) {
        for (int i = 0; i < num_tx; i++) {
            total_sent_packets += 1;
            total_sent_bytes += rte_pktmbuf_pkt_len(mbufs[i]);
        }
    }
}

// 计算数据包延迟
void calculate_packet_latency() {
    struct rte_mbuf *mbufs[BURST_SIZE];
    uint64_t current_time, tx_time;
    uint64_t latencies[BURST_SIZE] = {0};

    // 接收数据包
    int num_rx = rte_eth_rx_burst(PORT_ID, 0, mbufs, BURST_SIZE);
    if (num_rx <= 0) {
        return;
    }

    // 计算延迟
    current_time = rte_get_tsc_cycles();
    for (int i = 0; i < num_rx; i++) {
        tx_time = *((uint64_t *)rte_pktmbuf_extbuf(mbufs[i]));
        latencies[i] = current_time - tx_time;
    }

    // 打印延迟
    for (int i = 0; i < num_rx; i++) {
        printf("Packet %d latency: %" PRIu64 " cycles\n", i, latencies[i]);
    }
}
void calculate_traffic() {
    printf("Total sent packets: %" PRIu64 "\n", total_sent_packets);
    printf("Total sent bytes: %" PRIu64 "\n", total_sent_bytes);
}
void signal_handler(int signum) {
    if (signum == SIGINT || signum == SIGTERM) {
        force_quit = true;
    }
}

int main(int argc, char *argv[]) {
    init_dpdk();
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    struct rte_mempool *mpool;
    mpool = rte_pktmbuf_pool_create("mbuf_pool", BURST_SIZE, 32, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
    if (!mpool) {
        rte_exit(EXIT_FAILURE, "Error creating mbuf pool\n");
    }

    while (!force_quit) {
        uint64_t start_time = rte_get_tsc_cycles();

        send_packets_with_timestamps(mpool);
        calculate_packet_latency();
        calculate_traffic();

        uint64_t elapsed_cycles = rte_get_tsc_cycles() - start_time;
        uint64_t elapsed_us = rte_get_timer_hz() * elapsed_cycles / rte_get_tsc_hz();

        if (elapsed_us < DESIRED_PACKET_DELAY_US) {
            rte_delay_us(DESIRED_PACKET_DELAY_US - elapsed_us);
        }
    }

    rte_pktmbuf_pool_free(mpool);
    rte_eth_dev_stop(PORT_ID);
    rte_eal_cleanup();
    return 0;
}
