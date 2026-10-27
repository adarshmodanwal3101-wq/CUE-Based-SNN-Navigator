/* ============================================================================
 *  bridge.c - Icarus Verilog VPI module: TCP bridge to MATLAB (tcpclient)
 *
 *  $bridge_open();                                  listen on 127.0.0.1:5000, block until MATLAB connects
 *  $bridge_read(ok, a1,...,a10);                    read 10 bytes -> 10 integers, ok=1 (ok=0 on disconnect)
 *  $bridge_write(byte);                             send 1 byte
 *  $bridge_close();
 *
 *  Build (Linux/Mac):  iverilog-vpi bridge.c
 *  Build (Windows)  :  iverilog-vpi bridge.c -lws2_32
 *  Produces bridge.vpi  ->  run with:  vvp -M. -bridge snn_sim.vvp
 * ========================================================================== */
#include <vpi_user.h>
#include <stdio.h>
#include <string.h>

#define BRIDGE_PORT 5000
#define NBYTES_IN   10

#ifdef _WIN32
  #include <winsock2.h>
  typedef SOCKET sock_t;
  #define CLOSESOCK closesocket
#else
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <netinet/tcp.h>
  #include <arpa/inet.h>
  #include <unistd.h>
  typedef int sock_t;
  #define INVALID_SOCKET (-1)
  #define CLOSESOCK close
#endif

static sock_t srv = INVALID_SOCKET, cli = INVALID_SOCKET;

static int recv_all(unsigned char *buf, int n) {
    int got = 0;
    while (got < n) {
        int r = recv(cli, (char *)buf + got, n - got, 0);
        if (r <= 0) return 0;
        got += r;
    }
    return 1;
}

static void put_int(vpiHandle h, int val) {
    s_vpi_value v;
    v.format = vpiIntVal;
    v.value.integer = val;
    vpi_put_value(h, &v, NULL, vpiNoDelay);
}

static PLI_INT32 bridge_open_calltf(PLI_BYTE8 *u) {
    struct sockaddr_in a;
    int one = 1;
    (void)u;
#ifdef _WIN32
    { WSADATA w; WSAStartup(MAKEWORD(2, 2), &w); }
#endif
    srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv == INVALID_SOCKET) { vpi_printf("[bridge] socket() failed\n"); return 0; }
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, (const char *)&one, sizeof(one));
    memset(&a, 0, sizeof(a));
    a.sin_family      = AF_INET;
    a.sin_addr.s_addr = inet_addr("127.0.0.1");
    a.sin_port        = htons(BRIDGE_PORT);
    if (bind(srv, (struct sockaddr *)&a, sizeof(a)) != 0 || listen(srv, 1) != 0) {
        vpi_printf("[bridge] bind/listen on port %d failed\n", BRIDGE_PORT);
        return 0;
    }
    vpi_printf("[bridge] waiting for MATLAB on 127.0.0.1:%d ...\n", BRIDGE_PORT);
    cli = accept(srv, NULL, NULL);
    if (cli == INVALID_SOCKET) { vpi_printf("[bridge] accept() failed\n"); return 0; }
    setsockopt(cli, IPPROTO_TCP, TCP_NODELAY, (const char *)&one, sizeof(one));
    vpi_printf("[bridge] MATLAB connected.\n");
    return 0;
}

static PLI_INT32 bridge_read_calltf(PLI_BYTE8 *u) {
    vpiHandle call = vpi_handle(vpiSysTfCall, NULL);
    vpiHandle it   = vpi_iterate(vpiArgument, call);
    vpiHandle args[1 + NBYTES_IN];
    unsigned char buf[NBYTES_IN];
    int i, ok = 0;
    (void)u;
    for (i = 0; i < 1 + NBYTES_IN; i++) args[i] = vpi_scan(it);   /* ok + 9 values */
    memset(buf, 0, sizeof(buf));
    if (cli != INVALID_SOCKET) ok = recv_all(buf, NBYTES_IN);
    put_int(args[0], ok);
    for (i = 0; i < NBYTES_IN; i++) put_int(args[1 + i], buf[i]);
    return 0;
}

static PLI_INT32 bridge_write_calltf(PLI_BYTE8 *u) {
    vpiHandle call = vpi_handle(vpiSysTfCall, NULL);
    vpiHandle it   = vpi_iterate(vpiArgument, call);
    vpiHandle arg  = vpi_scan(it);
    s_vpi_value v;
    unsigned char b;
    (void)u;
    v.format = vpiIntVal;
    vpi_get_value(arg, &v);
    b = (unsigned char)(v.value.integer & 0xFF);
    vpi_free_object(it);
    if (cli != INVALID_SOCKET) send(cli, (const char *)&b, 1, 0);
    return 0;
}

static PLI_INT32 bridge_close_calltf(PLI_BYTE8 *u) {
    (void)u;
    if (cli != INVALID_SOCKET) CLOSESOCK(cli);
    if (srv != INVALID_SOCKET) CLOSESOCK(srv);
    cli = srv = INVALID_SOCKET;
#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}

static void reg_task(const char *name, PLI_INT32 (*f)(PLI_BYTE8 *)) {
    s_vpi_systf_data t;
    memset(&t, 0, sizeof(t));
    t.type   = vpiSysTask;
    t.tfname = (PLI_BYTE8 *)name;
    t.calltf = f;
    vpi_register_systf(&t);
}

static void register_all(void) {
    reg_task("$bridge_open",  bridge_open_calltf);
    reg_task("$bridge_read",  bridge_read_calltf);
    reg_task("$bridge_write", bridge_write_calltf);
    reg_task("$bridge_close", bridge_close_calltf);
}

void (*vlog_startup_routines[])(void) = { register_all, 0 };