// Automatically generated header file
// Date: 2024-12-15 09:58:02.679085
// Quantized model exported from octav_FCMNIST_Aug_BitMnist_4bitsym_width64_64_64_epochs60.pth
// Generated by exportquant.py

#include <stdint.h>

#ifndef BITNETMCU_MODEL_H
#define BITNETMCU_MODEL_H

// Number of layers
#define NUM_LAYERS 4

// Maximum number of activations per layer
#define MAX_N_ACTIVATIONS 128

// Layer: L3
// QuantType: 4bitsym
#define L3_active
#define L3_bitperweight 4
#define L3_incoming_weights 256
#define L3_outgoing_weights 64
const uint32_t L3_weights[] = {
	0x99888888,0x88889898,0x99811008,0x88988899,0x98012000,0x08899988,0x81332088,0x9988aa89,
	0x81430899,0x8a88aa88,0x023309a8,0x08088999,0x81230ca8,0x08008a99,0x11238bb8,0x00888899,
	0x10121999,0x80080088,0x11011099,0xa8800899,0x80010808,0x88010099,0x01111888,0x88800888,
	0x01100809,0x98880088,0x80121008,0x00980899,0x88122110,0x10000899,0x98801211,0x00110889,
	0x89889800,0x00888889,0x99980122,0x01180898,0x99800220,0x89888888,0x88012329,0xa9800088,
	0x98012419,0x99800889,0x9008232a,0xb8808889,0x8088125b,0xc9888009,0x88902368,0xb9888800,
	0x98002348,0x90888900,0x98801128,0x99888880,0x98808100,0x88988888,0x99980111,0x10880088,
	0x98888080,0x00000888,0x98088800,0x00800009,0x98888008,0x88800089,0x99990888,0x00800899,
	0x01108880,0x88001100,0x10109a90,0x00110110,0x0009cb00,0x00080001,0x018bd921,0x99880001,
	0x108cf942,0x9a800001,0x009cf843,0xa8000800,0x009dc833,0x99000811,0x188b9832,0x88880111,
	0x088a9011,0x80001101,0x88888001,0x80100000,0x08888808,0x00089890,0x00889988,0x08888880,
	0x18889880,0x01099980,0x80800888,0x80089801,0x00088888,0x01088801,0x10880000,0x80000111,
	0x88888800,0x08808888,0x88880088,0x11110088,0x08808098,0x01111008,0x88011180,0x11101800,
	0x80121100,0x10011080,0x80121188,0x08011888,0x00010108,0x80018899,0x81800800,0x22210898,
	0x80080801,0x12219998,0x90000011,0x1199a988,0x88888110,0x8aaa9800,0x8898808b,0xaaa99000,
	0x88889899,0x99890888,0x90080990,0x08808889,0x88808889,0x88989998,0x88808888,0x89899998,
	0x9889a999,0x99998899,0x89988888,0x88899989,0x88880089,0x80009989,0x88001189,0x01008999,
	0x80112200,0x00100899,0x81112322,0x10088998,0x80108808,0x80898088,0x88898aa0,0x20800108,
	0x9a998112,0x21222118,0x99900122,0x11111108,0x88888982,0x10001088,0x80898000,0x18000808,
	0x88000810,0x08088008,0x90000010,0x08088808,0x88011111,0x00008889,0x89800108,0x08098889,
	0x88808800,0x00008888,0x98801110,0x11080098,0x88023421,0x11100808,0x88123454,0x31000888,
	0x88110108,0x00089888,0x880818df,0xea888988,0x80808bdb,0x08808008,0x9899aa81,0x18908008,
	0x88888100,0x08898888,0x88088008,0x88988880,0x00001108,0x80809888,0x80880000,0x00010880,
	0x88890889,0x88008100,0x89988888,0x00008000,0x98898809,0x98808808,0x08988888,0x89880888,
	0x88888000,0x80008808,0x88888000,0x00000008,0x88800088,0x00111008,0x88011100,0x11121000,
	0x08812234,0x44220001,0x80001233,0x42100800,0x0880000b,0xec980800,0x8001809d,0xc9888890,
	0x08808089,0x99880088,0x88880800,0x88888880,0x98808000,0x80808888,0x88908810,0x88088998,
	0x88888188,0x80899888,0x88809990,0x88089900,0x08888880,0x09009888,0x88888889,0x89888888,
	0x11001080,0x08800011,0x01808889,0x9aa98800,0x0089abbb,0xacca8900,0x008abccb,0xcc988980,
	0x0089abed,0xda001180,0x10008801,0x35221011,0x18998223,0x31121111,0x1898118a,0xaaa81121,
	0x00008089,0x08008800,0x00880081,0x00088898,0x10898880,0x88898998,0x00081080,0x00899800,
	0x00810108,0x88801011,0x00008100,0x00001112,0x11000001,0x00001221,0x11100001,0x00001111,
	0x00080000,0x00000001,0x00088898,0x99999000,0x1089bcdd,0xedbaa980,0x0899aade,0xdcbcba98,
	0x10080080,0x09abaaa8,0x08080113,0x31100998,0x11011101,0x32111088,0x11000088,0x10088000,
	0x10009888,0x88888000,0x11000088,0x98800101,0x11008001,0x08000000,0x00008001,0x00080800,
	0x08088889,0x80088881,0x10889989,0x88998001,0x11101888,0x80800011,0x01112232,0x32101110,
	0x88888888,0x80888888,0x89888888,0x88888889,0x88888898,0x00880888,0x88808018,0x08080008,
	0x88000000,0x00888888,0x80080000,0x00000000,0x88888080,0x00008000,0x88888880,0x80818980,
	0x80098880,0x01800888,0x88010881,0x18888800,0x88011123,0x19a88000,0x899988ab,0xcda88010,
	0x89accdff,0xfb080020,0x899abba9,0x21000118,0x00121234,0x33221208,0x88123444,0x33211188,
	0x88888800,0x01222088,0x88899100,0x01443118,0x889a0311,0x24444310,0x899a2211,0x11244421,
	0x8999010b,0xedcb0122,0x880a9888,0x089b9012,0x8889a902,0x12189801,0x88899888,0x90888888,
	0x80800880,0x80188988,0x88000880,0x00008888,0x80889008,0x88800988,0x88889980,0x10888980,
	0x88988008,0x88088900,0x00998888,0x00a80008,0x88088898,0x89080008,0x80888988,0x88808088,
	0x88888888,0x00808000,0x80888808,0x00880808,0x88011088,0x08810888,0x00100000,0x80880808,
	0x00880008,0x88998008,0x88088088,0x80080121,0x88110010,0x88012232,0x89080809,0x90021232,
	0x9a9a9ace,0x13121110,0x9abbbbcf,0x23211000,0x9a9999bc,0x13120008,0x9a80888a,0x00111008,
	0x99988aa9,0x81112108,0x80888888,0x11131088,0x88899888,0x80210888,0x08888800,0x00180888,
	0x01008080,0x88080001,0x00008088,0x88888808,0x00080808,0x89aaa980,0x00808080,0x88880210,
	0x00080811,0x23664432,0x08880088,0x88001321,0x0808888a,0xbefffdb9,0x08880881,0x118bcdda,
	0x00088980,0x12130aaa,0x80880188,0x81111099,0x08088880,0x88011180,0x00808800,0x08818110,
	0x08888000,0x10088010,0x00008880,0x08888000,0x00088800,0x80088010,0x00080080,0x00000000,
	0x01008089,0x99980800,0x00008099,0x89000001,0x00808089,0x08002211,0x18080801,0x14444311,
	0x00880081,0x1189aab9,0x00008899,0xcfffffca,0x08088000,0x0009acba,0x80880888,0x02333188,
	0x00800009,0x88811111,0x00008901,0x08008020,0x00088800,0x80080810,0x08808000,0x00888800,
	0x08800800,0x00008900,0x08008008,0x88889880,0x00088080,0x00888000,0x00110000,0x08008000,
	0x08888888,0x08888888,0x89899998,0x01088888,0x88999989,0x08009989,0x88880088,0x00010989,
	0x80800882,0x21010999,0x88080010,0x08888899,0x011100a9,0x90008099,0x02312000,0x00009999,
	0x12222328,0x8088aaa9,0x81111109,0x8088a999,0x888089aa,0x81211000,0x8890999b,0xa0212331,
	0x8998908a,0x98012431,0x889aa9ab,0x98002331,0x88989ab9,0x10111110,0x08808980,0x11001088,
	0x00000000,0x00800110,0x00008000,0x80008000,0x00088800,0x00809000,0x00000808,0x80880000,
	0x00808080,0x80100880,0x00880010,0x08080800,0x00000990,0x88080001,0x08022118,0x00000000,
	0x89a82453,0x10080080,0x8affffa8,0x18000880,0x88bfffff,0xa9988980,0x00123221,0x08888088,
	0x00323210,0x08808080,0x00110809,0x08080080,0x00899880,0x00888801,0x00088908,0x88808000,
	0x00080000,0x00008000,0x00000008,0x88080000,0x01188008,0x00800000,0x00800800,0x08800800,
	0x01888090,0x80008880,0x00808000,0x00808880,0x00800008,0x08080000,0x00009998,0x08808000,
	0x00211108,0x00808080,0x08025653,0x00080008,0x0adffdb8,0x09988800,0x8adffffb,0x88888880,
	0x09889823,0x10088880,0x00143321,0x08800800,0x11111111,0x18808800,0x00000808,0x99890810,
	0x88888888,0x88888888,0x88888808,0x10888888,0x88999888,0x88009889,0x889a8888,0x99809888,
	0x80999899,0x88998999,0x88889888,0x98998aa8,0x81009988,0x98888a99,0x801089ba,0x98998808,
	0x0233128b,0x99880000,0x02344532,0x12222220,0x00112334,0x33332110,0x88888011,0x10210000,
	0x88980000,0x88000088,0x88888808,0x00888008,0x80888800,0x80888888,0x88888998,0x80080080,
	0x88800110,0x80008999,0x88801100,0x01100089,0x88881001,0x00000009,0x99a98980,0x00880008,
	0x99bbbbca,0x99880888,0x8aaabcb9,0x98880989,0x89881121,0x08baaba8,0x80112221,0x08898800,
	0x01201101,0x22111332,0x00801088,0x12234342,0x88801080,0x08122331,0x88888808,0x99980220,
	0x99998889,0x99990000,0x99980888,0x00088808,0x898889a8,0x08808088,0x89888808,0x08888889,
	0x08000080,0x08880080,0x00088888,0x99998008,0x00009900,0x00980000,0x08889900,0x00808800,
	0x80808980,0x89899810,0x08089888,0x81100120,0x0080889a,0x88011121,0x00880889,0xcdb99888,
	0x01088012,0x0accdcb9,0x12108014,0x4109bcba,0x12211123,0x53118baa,0x01210890,0x22110899,
	0x00880889,0x80011888,0x80898000,0x10011800,0x00000000,0x00010000,0x80000000,0x01000108,
	0x10010009,0xaaa88011,0x01111109,0xbdec9800,0x11111211,0x1aefb900,0x11800012,0x339dec80,
	0x00898000,0x342add80,0x1098899a,0x0320cd90,0x00808889,0x82219da0,0x10008008,0x91118b90,
	0x20800008,0x90008888,0x11000888,0x90008880,0x10008808,0x80000008,0x11008888,0x08808800,
	0x11000880,0x80880000,0x01008808,0x88001080,0x01110088,0x00800000,0x01000880,0x81801001,
	0x98980010,0x189a9999,0x99900133,0x33099a99,0x88811111,0x444099a9,0x888009a8,0x24530899,
	0x99989999,0x03553119,0x8999899a,0xa9024210,0x88998088,0x99901230,0x80088808,0x88880110,
	0x80180100,0x89881008,0x00000000,0x88988888,0x88088888,0x88888880,0x80888881,0x88888888,
	0x00809800,0x08889889,0x88988088,0x88888088,0x88000000,0x01808888,0x88800800,0x08889898,
	0x11111224,0x32188800,0x11010013,0x42088800,0x10089903,0x42888800,0x10898090,0x21a80990,
	0x008a9990,0x31888a90,0x18889909,0x32888aa0,0x00899888,0x118908a8,0x10800188,0x11881890,
	0x00801319,0x02180890,0x10901208,0x01088900,0x08900189,0x80898880,0x08998080,0x88908988,
	0x00900001,0x11009a80,0x00880198,0x08808800,0x11010000,0x08911001,0x11118888,0x01000101,
	0x00111109,0xaa801000,0x0011108a,0xdda90100,0x11211100,0xbec91111,0x11111018,0xacea8011,
	0x11108822,0x0dfc8010,0x21008022,0x2bfe9811,0x11888080,0x38beb901,0x10000091,0x309bb801,
	0x08000880,0x20809100,0x08808890,0x28880080,0x00008090,0x10808800,0x08880009,0x01880080,
	0x18808009,0x00008080,0x00888808,0x00088800,0x00880008,0x88100800,0x10080088,0x00100000,
	0x00000000,0x08000011,0x00008088,0x80888000,0x00008099,0x08800111,0x00800010,0x00800080,
	0x01890001,0x80090000,0x00888889,0x80011111,0x08000008,0x80111000,0x00911110,0x20800800,
	0x08812199,0x08888880,0x08abaca8,0x18aa8880,0x089acb24,0x19a98800,0x00110243,0xbdda9900,
	0x01100238,0xdba98080,0x0212110a,0xdb888000,0x00011189,0xca880000,0x10000109,0xaa998000,
	0x11011011,0x11010011,0x11111100,0x00800011,0x11111100,0x00088011,0x10100121,0x00880001,
	0x01008101,0x98080100,0x11110010,0x22108888,0x11000800,0x12208998,0x11998889,0x81228998,
	0x10809800,0x80338998,0x08800810,0x02429988,0x08890089,0x011adba8,0x08898aaa,0x9adeec98,
	0x08880898,0x9abbaa90,0x08990808,0x00089980,0x18889988,0x80088801,0x10808088,0x08088001,
	0x00880000,0x80001000,0x80000080,0x00000000,0x00008990,0x01188800,0x00080090,0x80089998,
	0x80000188,0x00888988,0x00000089,0x01089889,0x08008009,0x80089998,0x0800201a,0x91189998,
	0x8800011c,0xa1088000,0x8880809e,0x02100000,0x89aabcda,0x52188080,0x88aabb82,0x31010800,
	0x0899a812,0x21109001,0x08999111,0x11000000,0x08890110,0x10100010,0x00001110,0x10100100,
	0x88889898,0x89888089,0x989999a9,0x99a00008,0x89988998,0x88890100,0x89808980,0x08008110,
	0x88888808,0x88821320,0x90000099,0x8a932210,0x88008098,0x0da32200,0x88000880,0x0f942110,
	0x80801091,0x9f142100,0x88088080,0xab321008,0x88898089,0xa1311088,0x88000808,0x91111000,
	0x88008008,0x00010801,0x80808000,0x80980100,0x88088188,0x88880109,0x88801000,0x08080088,
	0x21233332,0x12221011,0x11222000,0x80100112,0x11008888,0x00000001,0x10080800,0x80880080,
	0x10880880,0x00008800,0x18800000,0x00080800,0x10000800,0x88000011,0x10100010,0x08888101,
	0x10001180,0x88888011,0x11000080,0x10008001,0x20800812,0x32110888,0x09980020,0x08888bb8,
	0x89baaabf,0xffeeddb8,0x08bdefff,0xffdcbb90,0x10099a99,0x98880011,0x21100001,0x11110112,
	0x00101001,0x11111000,0x11110001,0x00110111,0x01011101,0x08888800,0x11110988,0x9aa99880,
	0x11210988,0x99808988,0x11220180,0x08120000,0x11111208,0x13322211,0x01100088,0x21002220,
	0x08008011,0x88000110,0x08080008,0x9a000101,0x00898898,0x89008090,0x08898001,0x11100aa8,
	0x88998801,0x1200bcb9,0x08abaaab,0xbcddcca0,0x0099bcac,0xcdcb9880,0x10009989,0xa9988800,
	0x88898889,0x88889889,0x99888988,0x88008888,0x99880808,0x80008808,0x88888808,0x10888088,
	0x88888888,0x08888888,0x98808080,0x88808888,0x80800088,0x88800888,0x80008098,0x00808980,
	0x80080880,0x08898900,0x80900000,0x99808012,0x80888880,0x98800121,0x808888ab,0xb9001211,
	0x01108980,0x11233208,0x00124467,0x75311088,0x80123322,0x10089898,0x980089ab,0xbbaaaa98,
	0x10000010,0x08100000,0x00108880,0x01100001,0x10888888,0x11100001,0x00898880,0x00008001,
	0x00890088,0x89888888,0x08888888,0x80008898,0x08988888,0x98989998,0x00800880,0x09988a98,
	0x10880000,0x08888800,0x12108800,0x00080021,0x12210888,0x80000221,0x12422013,0x22222311,
	0x10123144,0x21121800,0x18888089,0x9aabaa98,0x089bcdff,0xfeccb900,0x00899bcb,0xcbb98000,
	0x80088808,0x00110800,0x08000001,0x01100008,0x00000088,0x00110080,0x80010898,0x10008898,
	0x80021188,0x01800080,0x80112211,0x00808900,0x08801200,0x11088808,0x88881188,0x088aca88,
	0x80008ab9,0x000cd988,0x80010099,0x00afc998,0x88001209,0x09cda998,0x88898890,0x18baa988,
	0x008a8a98,0x08998990,0x00999980,0x88089980,0x88800011,0x80189880,0x08801000,0x00008888,
	0x88888888,0x88088088,0x98880001,0x08010888,0x88800100,0x80021088,0x98001888,0x00100000,
	0x80001080,0x88008980,0x88800000,0x22208880,0x88888008,0x02210008,0x0899099a,0x98121210,
	0x000999a9,0xba021228,0x80080089,0xad921220,0x88990100,0x9d031110,0x99989980,0xab231088,
	0x9aa89888,0x99221888,0x9899899a,0x81221089,0x88989999,0x00111089,0x89889a98,0x00001889,
	0x11000888,0x88880000,0x10088999,0x99989800,0x10808989,0x08880980,0x00089990,0x80808888,
	0x00888889,0x88810001,0x100008a9,0x99811122,0x10080888,0x00099011,0x00880024,0x31098a90,
	0x0009aa82,0x21108a98,0x11218cfc,0x90200998,0x01244288,0x81288898,0x88012222,0x02288898,
	0x89a99900,0x11198980,0x89999001,0x10099800,0x00881111,0x00808800,0x10101000,0x01211100,
	0x88889998,0x88800088,0x88889888,0x00000008,0x98888008,0x01200088,0x89800111,0x12311000,
	0x88812180,0x01121080,0x80822109,0x99800900,0x98001011,0x08100888,0x88998023,0x10000999,
	0x89aaba98,0x99999999,0x998abbda,0x088a9099,0x88800099,0x80888988,0x88000008,0x88088800,
	0x88110010,0x88800008,0x80001808,0x80800089,0x88980888,0x88010888,0x88889989,0x00808888,
	0x88000111,0x00088800,0x08011322,0x10000808,0x88012112,0x11008000,0x08811188,0x00000088,
	0x080889aa,0x98888898,0x08110022,0x21008998,0x89802433,0x18808880,0x8aaa9890,0x08880100,
	0x89adffff,0xb8880100,0x00898999,0xa9800000,0x00121342,0x01080008,0x01111080,0x08889980,
	0x88080888,0x0889a988,0x08888900,0x88899988,0x08800889,0x00000888,0x00808008,0x00800888,
	0x10001122,0x20011000,0x01000121,0x18810110,0x10088012,0x8a812101,0x00988010,0xa8010010,
	0x08989120,0xb9109001,0x89898021,0x82089811,0x00001113,0x11000022,0x08110011,0x10012222,
	0x00100081,0x11111200,0x00000801,0x21880980,0x08889a99,0x9ababb98,0x889aabbb,0x9aaabb98,
	0x888aabab,0x99888888,0x0889baa9,0x99981000,0x0089a99b,0xabaa8001,0x00088999,0x9a980810,
	0x00801220,0x9a988000,0x8801342a,0xdb801000,0x0012440f,0xd0011100,0x880245bf,0xb1000001,
	0x808025bf,0x02000801,0x0888169f,0x22808000,0x8899829b,0x30888880,0x08998090,0x18808888,
	0x80889001,0x00008888,0x88888810,0x99000008,0x00888010,0x80080000,0x00080808,0x80000088,
	0x88888800,0x08080088,0x08888008,0x08988888,0x88888880,0x80890000,0x08080888,0x80081888,
	0x0800089a,0x01101800,0x80110aea,0x24200880,0x00118df8,0x53099980,0x11220ef0,0x62bba000,
	0x01222bf1,0x59c80111,0x011248d1,0x5ba01110,0x008010a1,0x2a010808,0x00980181,0x08000800,
	0x00008080,0x80088888,0x88888880,0x00800008,0x00088080,0x08000080,0x88088880,0x00880808,
	0x08888888,0x80080888,0x88880008,0x88008800,0x80008808,0x80880800,0x00880900,0x08888008,
	0x08888880,0x88888080,0x08099811,0x18988800,0x88808801,0x09880888,0x00099001,0x89889a98,
	0x00000812,0x9b999998,0x80088921,0xa98a9808,0x88809900,0x99899888,0x88998889,0x98888888,
	0x08a9a988,0x98888998,0x8880889a,0x80888808,0x00111000,0x33222331,0x10001121,0x33343432,
	0x00000088,0x88121321,0x80801018,0xaab98000,0x00081010,0x89ba9888,0x00888080,0x08999800,
	0x99a99998,0x08088899,0x99999900,0x08088099,0x99989901,0x00100888,0x99008808,0x88888888,
	0x98010008,0x80888008,0x90112219,0x88988889,0x8112239b,0x10989888,0x010133fc,0x28099880,
	0x022242f9,0x18088008,0x011131d8,0x08880810,0x01211098,0x89888800,0x81211200,0x98098808,
	0x80220018,0x08099888,0x88100088,0x00808898,0x98001088,0x88800088,0x98808880,0x88888899,
	0x11110109,0xa9980110,0x0101089b,0xdb800001,0x001109ce,0xa8100110,0x11118afd,0x81110000,
	0x01119cf9,0x12108980,0x1200bec1,0x3110a908,0x1109dc83,0x29998898,0x00999822,0x8a898980,
	0x00080110,0x80000000,0x00802098,0x00880100,0x18000881,0x18800081,0x00008001,0x18008801,
	0x00988800,0x00811001,0x08088889,0x00088011,0x10008800,0x00800111,0x01000008,0x08010110,
	0x00088890,0x81210800,0x8008809b,0x92520888,0x0000809d,0x94528980,0x0000018e,0x9440bc98,
	0x0001800f,0x962adba0,0x8010900d,0x860cb801,0x0088811a,0x039b8121,0x00110809,0x81881011,
	0x88000018,0x80801800,0x00888888,0x80080888,0x00888088,0x08880080,0x00808098,0x01008888,
	0x00808888,0x00088888,0x88800088,0x88080880,0x00808080,0x88098008,0x00880000,0x08888800,
	0x08888888,0x81110008,0x08888998,0xa0222108,0x08800009,0xa9123210,0x80001000,0xbc344388,
	0x00000001,0xbd443099,0x00008802,0xdc632aca,0x80088001,0xe051aba9,0x00080080,0xa129a988,
	0x08000008,0x81098800,0x08880880,0x80881000,0x80080808,0x00000000,0x80000808,0x08000880,
	0x00008800,0x08800888,0x00888888,0x00888899,0x88000880,0x00898888,0x88880088,0x88008888,
	0x00001111,0x10800000,0x00102111,0x11088801,0x00022100,0x11008880,0x01221800,0x11800988,
	0x00320118,0x01109aa9,0x01221180,0x341bdbba,0x01208813,0x40dfddba,0x08101232,0x8cdb8a99,
	0x08011189,0xaa911009,0x098008aa,0x90000080,0x9908aaa8,0x00000100,0x08899881,0x80001108,
	0x08098889,0x98000008,0x08088899,0x88080888,0x00880018,0x98a99888,0x08808808,0x00898000,
	0x00000000,0x08800000,0x00880011,0x11101110,0x88980122,0x20010011,0x08990121,0x11010200,
	0x08a98121,0x10110110,0x0aba8812,0x31221100,0x89cffca1,0x42108998,0x89abcdb0,0x319999a9,
	0x880009ba,0x09a99999,0x0021108a,0x89888088,0x00100888,0x88880808,0x00808880,0x90000188,
	0x88888888,0x99800088,0x08899880,0x80800008,0x80888108,0x80800800,0x00088800,0x00808800,
	0x08888999,0x89988888,0x80899999,0x8a999008,0x08889a80,0x08890010,0x88088a80,0x98080121,
	0x8808a988,0x80098111,0x888899b9,0x88990310,0x082211a9,0x89812321,0x01323441,0x01343100,
	0x01212211,0x12118a99,0x00101880,0x88abba89,0x80080a80,0x98a98088,0x80889800,0x01000888,
	0x88800888,0x88000898,0x88880008,0x00089998,0x88081010,0x01899898,0x88801101,0x11080888,
	0x08800000,0x80000000,0x00000088,0x80110008,0x00021088,0x88011100,0x01011100,0x08011010,
	0x00101080,0x10001010,0x81000088,0x00110100,0x01101199,0x22330000,0x00010898,0x3199aa99,
	0x88811112,0x8abddcb9,0x89a00238,0xdba9bbba,0x98aa9919,0xa8998ca9,0x08899988,0x08808ba9,
	0x01800820,0x10009b98,0x00080088,0x80889a80,0x00809088,0x90818898,0x00008889,0x88888888,
	0x00000800,0x02218008,0x808089aa,0x01210800,0x80888bcd,0xa0010100,0x080999de,0xda880008,
	0x001118cf,0xfa9a8890,0x1122121b,0xfca999a8,0x13323463,0x08889988,0x12208132,0x00088099,
	0x00099888,0x98008099,0x89980800,0x80000880,0x08980801,0x00080800,0x00808080,0x00100008,
	0x00000000,0x88808808,0x00000008,0x88808888,0x00110010,0x11000080,0x00000000,0x00010000,
	0x00100889,0x88888000,0x10000aa9,0x98898880,0x01212088,0x98009880,0x01243331,0x08000800,
	0x00233232,0x20888800,0x089cefed,0x80808888,0x9bfffc80,0x88008888,0x9a903432,0x89000108,
	0x08121008,0x80881000,0x0098a988,0x80000801,0x08888001,0x00880800,0x00888011,0x00088880,
	0x00808088,0x00008880,0x00810000,0x08888000,0x01000800,0x80000000,0x00088000,0x01000000,
	0x00000088,0x08000000,0x10008998,0x80880001,0x00809aaa,0x88108800,0x0100809a,0x90808980,
	0x11233332,0x10808080,0x10443542,0x10080090,0x8808adff,0x98080880,0x8adfff91,0x10808000,
	0x8aaa0332,0x00000101,0x80011088,0x88008110,0x00088088,0x90988088,0x10000008,0x88888888,
	0x10080801,0x00008980,0x10080088,0x80880800,0x10098008,0x81888080,0x11888888,0x08080001,
	0x00088899,0x99980800,0x00089999,0x99998080,0x000888a9,0x89888000,0x08800100,0x08801100,
	0x00801221,0x88080080,0x00800123,0x11110110,0x80889824,0x43110001,0x00089821,0x98008881,
	0x8001021a,0xd98a9880,0x8111118b,0xb9999880,0x00098989,0x99980008,0x11080881,0x00000888,
	0x11101111,0x80889898,0x11110000,0x88898880,0x00008080,0x80008088,0x00080800,0x00080808,
	0x80000080,0x08000088,0x08888888,0x88810108,0x88888008,0x9bb99008,0x808aaaa8,0x8cb89800,
	0x888abdca,0xdd800808,0x8889beef,0xfc088898,0x08800cee,0xa1210011,0x80098902,0x42100122,
	0x00898111,0x00888011,0x08980110,0x88808801,0x00098809,0x99908888,0x80088808,0x08008998,
	0x01100000,0x10800888,0x80000808,0x08080800,0x00080000,0x00011088,0x08000011,0x02211880,
	0x80000118,0x99988888,0x88011100,0x89999888,0x80122132,0x019a9890,0x00121021,0x008bb988,
	0x01100021,0x09cda889,0x00088125,0x0fdb8888,0x88088033,0xbe910010,0x88088012,0xba010110,
	0x88800800,0xa8801218,0x88800819,0xb8800088,0x88889809,0xca988999,0x88888009,0x8989a999,
	0x80080088,0x88889999,0x88010900,0x89888889,0x88801080,0x00989880,0x88800010,0x00888888,
	0x88988888,0x88888898,0x98880888,0x80998998,0x88800088,0x08898998,0x88808889,0x80088898,
	0x88001800,0x88881888,0x80888889,0x98888899,0x98888880,0x00888889,0x8899089a,0x88080009,
	0x8088089c,0xca9811aa,0x0088010a,0xabb9abba,0x11111110,0x9cccdeca,0x01122133,0x18aaaba8,
	0x01110102,0x43310880,0x81211100,0x13442320,0x01111110,0x12445318,0x88000012,0x22333188,
	0x88988888,0x98988888,0x88889988,0x00888898,0x89899898,0x00080889,0x888a99a9,0x00000888,
	0x89808898,0x10880808,0x88998899,0x08888000,0x8898808a,0x80808100,0x88100008,0x98000088,
	0x00010000,0x9a998998,0x08809810,0x8ab9ab88,0x00099001,0x09a99910,0x80898234,0x23222221,
	0x01008112,0x23445530,0x80088880,0x02234428,0x88899898,0x01101108,0x8888998a,0xb9980000,
	0x88000008,0x08080000,0x00810110,0x08088800,0x00011211,0x18000010,0x00001188,0x00800100,
	0x08800808,0x80080001,0x89980221,0x11808800,0x9adeddbb,0x00899901,0x9bdeeeba,0x80888081,
	0x89802331,0x81000011,0x01356538,0x01112131,0x0111209a,0x88000011,0x01000998,0x88889888,
	0x80108998,0x88988888,0x80808988,0x89888999,0x08808999,0x98899988,0x08888890,0x08088880,
	0x00008888,0x00080001,0x00808880,0x88001011,0x00080008,0x00000110,0x08088000,0x00800011,
	0x00088088,0x88000100,0x10111000,0x01000110,0x02322320,0x88088888,0x001238aa,0x9a99aba9,
	0x89bbdfd0,0x299aaaa9,0x9cffea03,0x21001008,0x8bba1121,0x00010110,0x88881210,0x80000121,
	0x80000131,0x00010120,0x08801000,0x00000800,0x00898000,0x00088800,0x00880888,0x89989800,
	0x00100888,0x01008000,0x00000880,0x10008080,0x00808081,0x08010980,0x00098881,0x10108901,
	0x00898980,0x90001801,0x10000a00,0x88811080,0x08810001,0x8a988aa9,0x00800132,0x9ccdccba,
	0x08881231,0x99abccb8,0x0808211a,0xa8809a90,0x0010110c,0xb8800888,0x0111021a,0xba900080,
	0x00001121,0x88988800,0x00008011,0x10018000,0x00000010,0x01008888,0x00008801,0x11080001,
	0x00808000,0x00000080,0x00088811,0x00808808,0x00088001,0x08800108,0x80088801,0x88880100,
	0x00008810,0x99000000,0x00080880,0x09888880,0x00088880,0x08a88981,0x08000888,0x18880801,
	0x08011088,0x29008001,0x800218b2,0x38008011,0x00009eb6,0x39800000,0x0089ef25,0x19998910,
	0x009aeb23,0x10089800,0x008ac901,0x22100100,0x0089bb80,0x22112100,0x08889998,0x00110000,
	0x989899a9,0xa9988888,0x88899aaa,0x9a888809,0x9999aa91,0x08881108,0x89999901,0x08080110,
	0x88899812,0x88080101,0x88999880,0x98800000,0x99880988,0x98889801,0x99801001,0x99980111,
	0x98800018,0x88811232,0x99088010,0x01080111,0x99890128,0x989a9880,0x88812429,0xa9999888,
	0x80112439,0xa9889889,0x80112431,0x8a980088,0x99012234,0x31111888,0x99981022,0x34310989,
	0x88088000,0x00888800,0x80088811,0x01008988,0x88008888,0x88000888,0x08899888,0x01008808,
	0x899aa888,0x00010800,0x89988009,0x90800008,0x8998010c,0x98011009,0x89989abf,0x8231008a,
	0x88999bbb,0x44308988,0x88808980,0x22199988,0x00008012,0x00888808,0x00000110,0x01011110,
	0x01001088,0x80001108,0x80000008,0x80880100,0x88001080,0x00800000,0x80801080,0x80808088,
	0x00001810,0x11011010,0x11000080,0x10008001,0x01108880,0x00889880,0x00080009,0x00009980,
	0x00881080,0x11118a98,0x00880108,0x8108ba98,0x00011100,0x318bbb90,0x00111001,0x30cca980,
	0x00889981,0x29ca8010,0x00898012,0x0cb80220,0x10000102,0xad821118,0x10110010,0x9a901880,
	0x00008888,0x0a998988,0x88088080,0x09a98980,0x00800101,0x19a98800,0x00808811,0x08988000,
}; //first channel is topmost bit

// Layer: L5
// QuantType: 4bitsym
#define L5_active
#define L5_bitperweight 4
#define L5_incoming_weights 64
#define L5_outgoing_weights 64
const uint32_t L5_weights[] = {
	0x32910009,0x01adca8b,0xa2ac5123,0x91022032,0x09910a8b,0x228a8aa1,0x09aa1da0,0x0ba08398,
	0x8021011a,0x0180a1ad,0xa10d028a,0xb3a15318,0x90a9a121,0x0988aa8b,0x2991aba9,0x5b821a99,
	0x44598131,0x112b5510,0x0a00a09a,0x30091918,0x23808000,0x034ce90a,0xa80a2a90,0x888191d9,
	0x098c8c13,0x28c9820a,0x8818911a,0x819a0110,0x922ada4c,0x90288a81,0x8099378a,0x888101ab,
	0x399c8908,0x993189af,0xb20133b9,0xa0110122,0xb8899348,0x112a0baa,0xa9082293,0x2902018a,
	0x9a810922,0x15108093,0x41001202,0xa2b04882,0x381802cc,0x898b9aa0,0x8008ba81,0x8ab0eb21,
	0x988a1980,0xaf1b32da,0xc0881912,0x10880436,0x04019b00,0xa2018111,0x8980102d,0xc9bac81a,
	0x0909889a,0x00088099,0x09980089,0x90809988,0x00888009,0x98990999,0x90890988,0x90800088,
	0xa0091888,0x02980094,0x78028880,0x0999f163,0x9880b980,0xd8a20219,0x20022b08,0xaa8bca0c,
	0x19980281,0x1781a910,0xaa0101e0,0xfa09bc59,0x99d9990b,0x81199881,0x01128083,0x2b12212b,
	0x980b1302,0x04122124,0x3111ba8a,0x9bada311,0xba8000b0,0x889aa819,0x88818399,0x181910af,
	0x89bb0188,0x28904408,0xba1c8880,0xcea9831a,0x93189aa0,0xa2a23213,0x95881211,0x108a0009,
	0x188a88b9,0x98b0a119,0x101a9c88,0x1891bb80,0x90abaab1,0x10029b93,0x9bb12311,0x1298122a,
	0x3318aafb,0xc0d311b9,0x9298bc01,0x21011210,0x098bb231,0x12212900,0x08a98220,0x88980018,
	0x0298a880,0x85396611,0x2830db89,0x001c1911,0x220849a8,0x1018d029,0x9122a100,0x01288a90,
	0x19021183,0x89809918,0x99aa3101,0x0b238900,0x0d98199a,0x081a8ad8,0x88889ac2,0x931d4331,
	0x8990d9ae,0xe84acc99,0x09482499,0x20812011,0x0122189b,0xc0b02212,0x884a01b8,0x10a8a891,
	0xb8d98d28,0xa3bb1290,0x19100323,0x219f2911,0x88a841e0,0x10b3240a,0xba2a827a,0x88809bdb,
	0xa1a110bc,0x119cab01,0x88811092,0x89f91023,0x0a91bcab,0x0ac82331,0x33820c10,0x0200ba92,
	0xd9b1b8df,0x88883392,0x00411220,0x00981188,0x991842a9,0xaab08922,0x84770331,0x20a00180,
	0xc890a012,0x890a8a28,0x511580aa,0xdcf009e8,0x8ca03198,0x89880212,0x0b819814,0x4289cb11,
	0x8ba01288,0x29b308bc,0x829280a0,0x0200c909,0xb8bb8018,0x3a90199e,0x93929d01,0x85430011,
	0x9db21322,0x2904339a,0xca0a9989,0x8018938b,0x08c00291,0x00c2109a,0x02228a81,0x98423010,
	0x9b9123a0,0x18118808,0x09208990,0xcb100113,0xa88019db,0x9e990090,0x921399b2,0x03491221,
	0x198809da,0x91860189,0x889ab999,0x91893299,0x9180d144,0x3810a82a,0x8999f880,0x81191850,
	0x8a100df9,0x98889aad,0xbab19021,0x18288888,0xa819fa41,0xac233b8b,0x1a1aa42a,0x02202522,
	0x92224b11,0x0b20ab9b,0xda888188,0xba0a1800,0x99838189,0xc98aa831,0x089c99a9,0x210a248b,
	0x3240989e,0xbc2a029a,0xd8290109,0x90e01343,0x18809031,0x0229b280,0x29a9219c,0x09a8aa9a,
	0x11800121,0x2819941a,0x00b881a9,0x0a9abab9,0x8cdc28aa,0x0099aba8,0x33311224,0x12911398,
	0x090a8b99,0xa9a49a38,0x1c830813,0x98310899,0x00909310,0x09021920,0xb3a901b3,0x04b2399a,
	0x09822324,0x1b002184,0x152399a8,0xa899fccc,0x990989a0,0x112899ea,0x0821b184,0x3b912101,
	0x390bb010,0x0389ba14,0x2ebf8881,0x10320140,0x18830080,0x28880108,0x21980114,0x3a1b8180,
	0x8a998311,0xa000a9b0,0xc20b2115,0x10013083,0x99114c9a,0x40380aab,0xb8020094,0x22288011,
	0x89a2b010,0x0b1030b0,0x03818103,0x321099ed,0x3d800099,0x03181181,0x8180881f,0xd2933014,
	0x820b922b,0x82400921,0x3b08f1b3,0xba988900,0x8aa99131,0x0140ab98,0x86990811,0x188030e1,
	0x9c88b000,0x8b001292,0x2d881221,0x11048ab8,0x09a02012,0x3111880a,0x0802045c,0x90824003,
	0xa9a90b21,0x5849980a,0x902b8983,0x11381118,0x928aa2ba,0x00bcceb1,0xa6532388,0x888080b9,
	0x08180235,0x2811992b,0xaaba3391,0x0124ba9a,0x0808c908,0x0411189b,0x1a88100a,0x0c0b1880,
	0xba102091,0x30129911,0x10904234,0x1800b888,0x22281abc,0x9b198000,0x2180a1b9,0x9aa8cc42,
	0x18a11208,0x00909101,0x89a813a2,0x0018b89a,0x88900abd,0x01aa0408,0x18443c20,0x129b8818,
	0x0fc9a812,0x89909830,0x10019812,0x19203009,0x08088815,0x11243901,0xab100222,0x030009a3,
	0x000ba929,0xab108930,0x18484311,0x20308a00,0x88918181,0x02102cb8,0xf9380889,0xa28829ba,
	0x92290208,0x08c01899,0x9880cfbe,0x18018aab,0x01191044,0x28a64a11,0x0304220a,0x92298800,
	0x08b9a212,0x8aaa9810,0x1801aebb,0xaa221a00,0xad800123,0x50143033,0x9b001814,0x48220903,
	0x8ca13b99,0x280cab99,0x90201000,0x989ca001,0x902100aa,0xcca22519,0x183208b8,0x000408a1,
	0xb3082f9b,0x88e1a8a8,0x90a08842,0x22a988aa,0x980b8147,0x3a821989,0x933a2958,0x80022888,
	0x20410099,0x90bbeda9,0x818990ac,0x10821890,0xab800394,0x15323a91,0xa8988aa1,0xc0a913c1,
	0x9880010c,0xc28891ac,0xb80aba12,0xa08c6101,0xedc8b310,0x2320012b,0x92818033,0x20211810,
	0x00821988,0x998b99ae,0xfdba8808,0xb9bd8200,0x08b1b080,0x8a989446,0x30ba0b29,0x04331118,
	0x90a19c0a,0xbdd1aa01,0x01080822,0x099c28b0,0x01891231,0x80c35420,0x09cb0100,0x01899c82,
	0x32029c8b,0x8892d922,0x1982b999,0x8001048a,0x12113122,0xc191a193,0x28888008,0x0ba81098,
	0x120220c9,0x83382343,0x19018889,0x0b89b218,0x889309aa,0xbb8883b2,0x2bdb8f9b,0x980818a8,
	0x81210811,0x00322212,0x101081ab,0x4824d989,0x01802900,0x184fbf98,0xd11892a8,0xa9af880a,
	0x081808a9,0x180cdd21,0x81213123,0xb9b1acc8,0xa0890020,0x03022988,0xc8018411,0x24acaa10,
	0x81301aaa,0x9aa0b814,0x14210a82,0x1002c8ba,0x1330b012,0x02132881,0x0acda891,0x0cc90881,
	0x88811889,0x08a0b923,0x411a8090,0x4424d0ac,0x81a8a388,0xdba08800,0x2190882f,0xe1800902,
	0x8819ae98,0xa21da884,0x24020200,0x1822099a,0x90deb011,0x80a34b18,0x328e0880,0x10ab8890,
	0x02980aa8,0xa18301d6,0x0e9319a9,0x28094214,0x81344c9a,0x0c998388,0xae88a98b,0x8a209090,
	0xcd99a828,0xd90918a9,0x33ccb831,0x94802000,0x84493920,0x19018941,0x8c088a00,0x139dbb2e,
	0x9882adbb,0x890882a9,0xc1bb99a9,0x11288900,0x53445189,0x0b9bb050,0xb99a2812,0x81280881,
	0x83003922,0x20003088,0x1a849aab,0x00ac1211,0x1910012a,0x180db109,0x828988a0,0xb128dab8,
	0x890ab032,0x008410a4,0x20a4028a,0x0ddf0010,0x2b38b081,0xa9010911,0x50802393,0x0a229a80,
	0x91221aaa,0xa9c23302,0x0150ab8b,0x109b1202,0x88180322,0x01018ba8,0x9ea9b9a0,0xc0101199,
	0x811103b8,0xcb84462c,0xc112a809,0x9c299212,0xabb0bb99,0x802880b8,0x8f01ca98,0x1283000b,
}; //first channel is topmost bit

// Layer: L7
// QuantType: 4bitsym
#define L7_active
#define L7_bitperweight 4
#define L7_incoming_weights 64
#define L7_outgoing_weights 64
const uint32_t L7_weights[] = {
	0x32128a18,0x000198a2,0x111ccab0,0x2352002b,0xd0120280,0x180b1808,0x1398a90d,0x08898ac8,
	0x0911a808,0x00082218,0x8b8b8a81,0x020801c2,0x10339829,0x41081313,0x11b9b989,0xb88c89aa,
	0x91012000,0x12311831,0x23242111,0x0cc010b0,0x0c391801,0xe39911a9,0xbcddbab9,0x81c08180,
	0xa11a1580,0xcdba1129,0x89c38298,0x34eb20fb,0x311899aa,0x39181a00,0x11098984,0x4a0913ad,
	0x9a1002b8,0x4288131a,0x81899881,0x09bb92a4,0x2941bb0b,0x22080281,0x90809440,0x0210019d,
	0x089eadb9,0x10d823b0,0x199a0910,0x91920020,0xaa8209af,0x2b320049,0x20293a11,0x0a810899,
	0xb9aeb998,0xcf092088,0x91984881,0x238b0499,0x39391918,0x191211b5,0x019a01a0,0xa89c1080,
	0x8134ac00,0x93922018,0xcfaab80b,0xcb90c0f0,0xb4381319,0x1028c228,0xd00d1019,0x80090a98,
	0x89a008a0,0x80aa9bd3,0xcb3b3211,0xbf99a897,0x309818a0,0x20b1abb1,0x3099219f,0x181282a0,
	0x0910eee8,0x8aa10222,0xcaa19b31,0xa0029091,0xa81823a9,0x20252142,0x1a8b0818,0x989c93a0,
	0xdcde1280,0x2890ae90,0x50150423,0x08b93019,0x9ab0a920,0xaa30148f,0xbb990bc1,0xb080039a,
	0x81012990,0xa99ba9a3,0x21032282,0x33210aa2,0x30880820,0xa89983a8,0x88ca01ba,0x8aba9d99,
	0xe0800888,0x08183022,0x24a62183,0x90a013a2,0x10a10029,0x42989acb,0x819b188a,0x1019c092,
	0x9b819080,0x12862888,0x3388a038,0x1a090a19,0xa1002231,0x888923db,0x8b020908,0xb2281129,
	0x18ea8080,0x8a88c000,0x2540c0a0,0xd98b2811,0x90422822,0x9998120b,0x10080801,0x9202bbb8,
	0x08c01388,0x22909118,0x25281bd8,0x829a8a11,0x09092038,0xaca918a8,0xa8a10820,0x9219a98a,
	0xeab01221,0x434320a1,0x24211802,0x2a0809b9,0x8da9b809,0xbdad109a,0xea9a88c8,0x43380981,
	0x22ad8190,0x91a99ee3,0x41083083,0x2b21291c,0x80aa0980,0xb9819f28,0x1211028a,0x81120110,
	0x9afcb188,0x991089b9,0xe12a9e23,0x0830201f,0xbea98089,0x911119d8,0x28040c19,0x22220003,
	0x110f8938,0x02d89099,0x45332801,0xa0920eab,0x8cac8c28,0xd2201814,0x831023b9,0x1ddf0181,
	0x991b0a09,0xba919a11,0xaa108301,0x394089b0,0x18c02811,0x8c098000,0x4c0b12c1,0x99393200,
	0x1a888aa0,0x8a1988a1,0x092099b0,0xc8183251,0x9b0b28b8,0x1881b918,0x2153132b,0x3241a019,
	0x8088a380,0x10302120,0xa2800891,0x191b2aca,0x19139a21,0x992a2292,0x98a2c8ba,0xb0815208,
	0x82299908,0x2320a02d,0x0aa20899,0x11cc88a9,0x0205b9a0,0x02118089,0xca81b808,0x644303a0,
	0x931298a0,0x9da10bcc,0x9a9ae139,0x9b219a31,0x00a8a010,0x00318829,0x11280533,0x0a23100c,
	0x8a141be8,0xdf001a08,0x0da3b20a,0x18989929,0x84911311,0x00880bcb,0x09912224,0x88981889,
	0x90c99400,0x1818802c,0x20818b2a,0x9e2c0ac4,0x3919e998,0x081280b0,0xa98aea0a,0x52493202,
	0xb90999c9,0xfc0b4090,0x01920202,0x0180443b,0xb4349018,0x032130d8,0x3890d808,0xbbab0aca,
	0x02190088,0x089ab801,0x80015321,0x2a9383a2,0x0b1030a1,0x82b8aaa3,0x0ab881ac,0x3aca012a,
	0x12222220,0xb590a388,0x89b98210,0x868180b8,0x112180b9,0x10091239,0x900dbeb8,0xcfbe9a11,
	0x099c9500,0xcd9ab309,0x18125801,0x8fa91d9c,0x283a0821,0xc900c810,0xb9208211,0x89ad8981,
	0x36821088,0xc92188bc,0xa9cab8b9,0x4341931c,0xac301018,0xaade0291,0x1103cc8b,0xacb93c92,
	0x04202108,0xe0a98218,0xcaac9adf,0x3b020819,0x1011a90d,0x98baa318,0x19181a20,0xbfb99a03,
	0x82a22010,0xa105248a,0xd1198898,0x32a28181,0x9fcbbbdf,0x1a100198,0xb80ba982,0xfc8e0313,
	0x31128281,0x9122210c,0xdca18a00,0x23181188,0xa49b91aa,0x9980ab89,0xb200ddb1,0xa0880431,
	0x10521910,0x0221420c,0xb9d88a89,0x30130808,0x9ac0899b,0x39381080,0xcba8c10c,0x8a298522,
	0x98b989b8,0x9189129a,0xacaa9191,0xaac98918,0x928b90ab,0x35229211,0x91100220,0x11b09c21,
	0x2119c261,0x11a21109,0x24318b9a,0x10159b0d,0xd999dd99,0xa99c0838,0x3a12f9ba,0x9b292109,
	0x29ba3009,0x90989299,0xa1021311,0x99890000,0x39abac9a,0x11221482,0x91299112,0xaab21050,
	0xab819bb8,0x99288190,0xac80c000,0x2138129c,0xabd89989,0xb841aa6b,0x19801008,0x12132240,
	0x9bb80a99,0xb99888a0,0x820fb998,0x34111211,0x9029b948,0x9a0b0801,0x4511a110,0x81139b01,
	0x18a89c08,0x111a31d9,0x98b981a1,0x88091010,0x10cc8b9f,0x8032813c,0x819a1321,0x21a9a82e,
	0x1a080181,0x23099981,0x1138a099,0x9001504d,0xac001028,0x8b918a88,0x999332a8,0xc8f89310,
	0xa2282348,0x21938508,0x08019181,0x1880fcb8,0x00d9fd88,0x099c11b0,0x8110ab00,0x2a312912,
	0x10000898,0xa8219821,0xaaa8a088,0x1203402b,0xa0ba8aba,0x0d809910,0x012299ac,0x91012616,
	0x81300110,0x190aba4b,0xa8a12000,0x82920880,0x09b8d9da,0x8ba098a3,0x9820f8a0,0x20193302,
	0x01318339,0x0a808129,0xaea2b8db,0x89882fc1,0x220219b3,0x099a9c09,0xa933d080,0x28304289,
	0x24191821,0x2d84038a,0x00b80e99,0x81190aa0,0x08029182,0x880b218c,0xda88c910,0x53313880,
	0x13133a20,0xd20191a1,0xa1201100,0x4152832a,0x0c9c4888,0x08ca8990,0x29a11a9d,0xbba88880,
	0x92242c00,0x98209219,0x0208b011,0x50319000,0x00008390,0x900dabb0,0x01aa99cc,0x93112b9a,
	0x2a801510,0x29222d91,0x33108ac8,0x88aa8cc0,0x22a21888,0x0998202a,0xa30aaac8,0x2818b8bb,
	0x31288448,0x02930980,0x99101000,0x08208caa,0x119828a0,0x2a8b0840,0x019b93da,0x9a9a189a,
	0x21a82110,0xb8f899ab,0xba8b8b1b,0xea18d1a2,0x331abc90,0x22919919,0x34228223,0x2c13bb0b,
	0xba1988c8,0xebbbd820,0xc81aadba,0x20912a3b,0xa3989b3b,0xba9a801a,0x2a123132,0x2902194a,
	0x33035109,0x80a33928,0xaaa88ec8,0x2112dda1,0x482b12a9,0x09811200,0x81eba22b,0xcfffa8aa,
	0x19101298,0x3988be80,0x9fb82ac9,0x88849828,0x388a0811,0xb9a8dd99,0xea901811,0xb3a21981,
	0x239b0100,0x198bbc90,0x2a8f9b9a,0xa00abba2,0x11411821,0xbbda0c29,0x80141812,0x80829ea1,
	0x8823ba88,0x1011583e,0xdf980909,0x00a80c18,0x18aa1b00,0xda89fdaa,0xb9382281,0x91123800,
	0x81438108,0x31401a0c,0xbeea14ca,0x381999a1,0x8112091c,0x9900b80d,0xac900889,0xa1431419,
	0x280128a8,0x89821b11,0x8c01b201,0xbb8ac081,0x022a5192,0x13218308,0x0a8b9800,0x0be81b88,
	0xb1bf0198,0xdabed9b0,0x32054151,0x88911810,0x10000901,0x8cb08090,0x101100b3,0x0eff8930,
	0x9a31b801,0x8112312c,0xaf815299,0x92004b92,0xa8110190,0xb801ffb8,0x8b209298,0x12819988,
	0x21332211,0x98900a90,0x2b90398c,0xb2089bf8,0x22242280,0xc21c0810,0x19a90880,0x9bbe099b,
	0x990a9990,0x88880808,0x88980090,0x89880098,0x00098999,0x89980889,0x88988809,0x0088809a,
}; //first channel is topmost bit

// Layer: L9
// QuantType: 4bitsym
#define L9_active
#define L9_bitperweight 4
#define L9_incoming_weights 64
#define L9_outgoing_weights 10
const uint32_t L9_weights[] = {
	0x3140c8c0,0xbef0aaa9,0x989ccccc,0xcfcf0424,0x18119cd0,0xa1decb10,0x12423058,0x11000248,
	0x24034548,0x33003ab0,0x9ec99a28,0x92741325,0x1ebad8aa,0x21cc01fc,0xbccacddf,0xbacdc888,
	0xa0201a10,0x51921ca0,0xc386118d,0xa9ad312d,0xcadc993c,0xcac9cb98,0x2881c8a2,0x9b4148c8,
	0xc91109c8,0x89401302,0x71888944,0xab19aab8,0xdacc898c,0xcb32d984,0x082009bc,0x8bc8b9b8,
	0xbabe23ab,0xc8b1c988,0xa89981bd,0x38089a89,0x80104813,0x25ca9ac9,0xbada444b,0xc8aa9aa8,
	0x99a10cac,0x98ba98da,0x0eb81014,0x9d898cba,0xb8238ba1,0xc8123142,0x91b809d0,0x0439c418,
	0x99b09820,0x9a8edfcd,0xdcbefdeb,0xbddb0323,0x56341428,0x89023300,0x199a9808,0xabaa99c8,
	0x8aada8a6,0x1452121b,0xb00b1a8a,0x2332ac99,0xb0a9081b,0xcbece8dc,0xeaacbbb0,0x82133240,
	0x30a1a8ab,0xf1b08880,0x9368341b,0xb8cccbcb,0xaaaa1494,0x0122129a,0x2302edd8,0xbbcabc88,
	0xb8ed9cac,0xbcfdc144,0x21b9809a,0x13300a0b,0x19ababd1,0x391b0aa8,0xa1898334,0x52019aa9,
}; //first channel is topmost bit

#endif
